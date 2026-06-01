"""
Solid temperature uniformity optimization using trained PINN.

Given a trained PINN model f(x, t, T_preheat, T_h2o2) → fields,
find optimal (T_preheat*, T_h2o2*) that minimize solid wall temperature
non-uniformity while maintaining sufficient sterilization temperature.

Usage:
    python optimize_uniformity.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import h5py

from config import DataConfig, PINNConfig, F_T
from pinn_model import PINNModel
from data_loader import load_cas, compute_cell_centers


def get_solid_coords(cas_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract solid cell centers and ALL cell centers from cas.h5 mesh.

    Returns:
        solid_coords: [n_solid, 3] raw coords of solid cells
        all_coords:   [n_total, 3] ALL cell centers (for coord normalization)
    """
    mesh = load_cas(cas_path)
    cell_centers = compute_cell_centers(mesh)
    n_total = len(cell_centers)
    print(f"  Total cells: {n_total}")

    with h5py.File(cas_path, "r") as f:
        zones = f["meshes/1/cells/zoneTopology"]
        zn_name = [n.tobytes().decode() if isinstance(n, bytes) else str(n)
                    for n in zones["name"]]
        zn_min = zones["minId"][:]
        zn_max = zones["maxId"][:]

    solid_mask = np.zeros(n_total, dtype=bool)
    for iz, name in enumerate(zn_name):
        if "soild" in name.lower() or "solid" in name.lower():
            solid_mask[int(zn_min[iz])-1:int(zn_max[iz])] = True

    solid_coords = cell_centers[solid_mask]
    fluid_coords = cell_centers[~solid_mask]

    # Match training data order: fluid first, then solid
    all_coords = np.concatenate([fluid_coords, solid_coords], axis=0)

    print(f"  Solid cells: {len(solid_coords)}")
    print(f"  Fluid cells: {len(fluid_coords)}")
    print(f"  Zones: {list(zip(zn_name, zn_min, zn_max))}")
    return solid_coords, all_coords


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data_cfg = DataConfig()
    pin_cfg = PINNConfig()

    # ── 1. Load mesh geometry ──────────────────────────────────
    print("\n[1/4] Loading mesh geometry...")
    solid_coords_raw, all_coords = get_solid_coords(data_cfg.cas_path)
    n_solid = len(solid_coords_raw)

    coord_min = all_coords.min(axis=0)
    coord_max = all_coords.max(axis=0)
    coord_range = coord_max - coord_min + 1e-12
    norm_solid = (solid_coords_raw - coord_min) / coord_range
    x_solid_all = torch.tensor(norm_solid, dtype=torch.float32)

    print(f"  Coord range: min={coord_min} max={coord_max}")

    # Temperature normalization
    T_min = 273.15
    T_max = 573.15
    T_range = T_max - T_min + 1e-12
    T_preheat_0 = pin_cfg.T_preheat_default
    T_h2o2_0 = pin_cfg.T_h2o2_default
    T_p0_norm = (T_preheat_0 - T_min) / T_range
    T_h0_norm = (T_h2o2_0 - T_min) / T_range

    print(f"  T range: [{T_min:.1f}, {T_max:.1f}] K")
    print(f"  Baseline: T_preheat={T_preheat_0:.1f}K  T_h2o2={T_h2o2_0:.1f}K")

    # ── 2. Load trained model ──────────────────────────────────
    print("\n[2/4] Loading trained model...")
    model_path = (Path(__file__).parent / "output" / "model_best.pt")
    if not model_path.exists():
        model_path = Path(__file__).parent / "output" / "model.pt"
    print(f"  File: {model_path}")

    model = PINNModel(
        n_spatial=3, hidden=pin_cfg.hidden_size,
        n_layers=pin_cfg.n_hidden_layers,
        activation=pin_cfg.activation,
        n_output=pin_cfg.n_output_fields,
        parametric_bc=pin_cfg.parametric_bc,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "net.0.weight" in state:
        model.load_state_dict(state)
    elif isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params:,} params")

    # ── 3. Sample points & baseline eval ───────────────────────
    print("\n[3/4] Evaluating baseline uniformity...")

    rng = np.random.default_rng(42)
    n_spatial = min(n_solid, 2000)
    idx = rng.choice(n_solid, size=n_spatial, replace=False)
    x_solid = x_solid_all[idx].to(device)

    n_time = 20
    t_norm = torch.linspace(0.0, 1.0, n_time, device=device)

    @torch.no_grad()
    def eval_solid_temp(tp_n, th_n):
        bc = torch.tensor([[tp_n, th_n]], dtype=torch.float32, device=device)
        T_all = []
        for t_i in t_norm:
            t_b = t_i.expand(n_spatial, 1)
            b_b = bc.expand(n_spatial, 2)
            pred = model(x_solid, t_b, b_b)
            T_all.append(pred[:, F_T:F_T+1])
        T_phys = torch.cat(T_all) * T_range + T_min
        return T_phys.std(), T_phys.mean(), T_phys.max(), T_phys.min()

    std0, mean0, max0, min0 = eval_solid_temp(T_p0_norm, T_h0_norm)

    print(f"  Baseline ({T_preheat_0:.0f}K / {T_h2o2_0:.0f}K):")
    print(f"    Mean(T)  = {mean0:.2f}K")
    print(f"    Std(T)   = {std0:.2f}K")
    print(f"    Max-Min  = {max0 - min0:.2f}K")

    # ── 4. Gradient optimization ───────────────────────────────
    print("\n[4/4] Optimizing T_preheat, T_h2o2...")

    T_p = torch.tensor([T_p0_norm], device=device, requires_grad=True)
    T_h = torch.tensor([T_h0_norm], device=device, requires_grad=True)

    # Target: uniform temperature at ~410K (sterilization requirement)
    target_mean = (T_preheat_0 + 5.0 - T_min) / (T_max - T_min)  # ~5K above baseline
    target_mean_K = target_mean * T_range + T_min
    print(f"  Target mean solid T: {target_mean_K:.1f}K")

    opt = torch.optim.Adam([T_p, T_h], lr=0.005)
    n_iter = 400
    w_std = 1.0
    w_target = 2.0   # penalty for deviating from target mean

    for i in range(n_iter):
        opt.zero_grad()
        bc = torch.cat([T_p, T_h], dim=-1)
        T_all = []
        for t_i in t_norm:
            t_b = t_i.expand(n_spatial, 1)
            b_b = bc.expand(n_spatial, 2)
            pred = model(x_solid, t_b, b_b)
            T_all.append(pred[:, F_T:F_T+1])
        T_phys = torch.cat(T_all) * T_range + T_min

        # minimize std + penalize deviation from target mean
        dev = (T_phys.mean() - target_mean_K) / T_range
        loss = w_std * T_phys.std() + w_target * (dev ** 2) * T_range
        loss.backward()
        opt.step()

        with torch.no_grad():
            T_p.clamp_(0.0, 1.0)
            T_h.clamp_(0.0, 1.0)

        if i % 80 == 0 or i == n_iter - 1:
            tp_K = T_p.item() * T_range + T_min
            th_K = T_h.item() * T_range + T_min
            print(f"  iter {i:3d}: T_preheat={tp_K:.1f}K  T_h2o2={th_K:.1f}K  "
                  f"std={T_phys.std():.2f}K  mean={T_phys.mean():.2f}K  loss={loss:.2f}")

    # ── Final comparison ───────────────────────────────────────
    tp_opt = T_p.detach().item()
    th_opt = T_h.detach().item()
    T_p_opt_K = tp_opt * T_range + T_min
    T_h_opt_K = th_opt * T_range + T_min

    std_opt, mean_opt, max_opt, min_opt = eval_solid_temp(tp_opt, th_opt)

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"  Baseline       T_preheat={T_preheat_0:.0f}K  T_h2o2={T_h2o2_0:.0f}K")
    print(f"  Optimized      T_preheat={T_p_opt_K:.1f}K  T_h2o2={T_h_opt_K:.1f}K")
    print(f"  Delta T_preheat:  {T_p_opt_K - T_preheat_0:+.1f}K")
    print(f"  Delta T_h2o2:     {T_h_opt_K - T_h2o2_0:+.1f}K")
    reduction = 100*(std0.item()-std_opt.item())/(std0.item()+1e-12)
    print(f"\n  Std(T):   {std0:.2f} → {std_opt:.2f} K  ({reduction:.1f}% reduction)")
    print(f"  Mean(T):  {mean0:.2f} → {mean_opt:.2f} K")
    print(f"  Max-Min:  {max0.item()-min0.item():.2f} → {max_opt.item()-min_opt.item():.2f} K")

    # Per-time breakdown
    print(f"\n  Per-time comparison (mean +/- std, K):")
    for i, t_i in enumerate(t_norm):
        bc0 = torch.tensor([[T_p0_norm, T_h0_norm]], device=device)
        bc1 = torch.tensor([[tp_opt, th_opt]], device=device)
        t_b = t_i.expand(n_spatial, 1)
        with torch.no_grad():
            T0 = model(x_solid, t_b, bc0.expand(n_spatial, 2))[:, F_T] * T_range + T_min
            T1 = model(x_solid, t_b, bc1.expand(n_spatial, 2))[:, F_T] * T_range + T_min
        if i % 4 == 0 or i == n_time - 1:
            print(f"  t={t_i.item():.2f}  base: {T0.mean():7.2f}+/-{T0.std():5.2f}  "
                  f"opt: {T1.mean():7.2f}+/-{T1.std():5.2f}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
