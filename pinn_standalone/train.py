"""
Multi-fluid PINN training script — 3-phase Eulerian-Eulerian + species transport.

Handles 20-field output (see config.py for field layout).
"""
import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset

from config import (DataConfig, PINNConfig, FIELD_NAMES, N_OUTPUT_FIELDS,
                     F_T, F_P, F_K, F_OMEGA, F_VOF2, F_VOF3,
                     F_Y_H2O2_V, F_Y_H2O_V, F_Y_AIR_V, F_Y_H2O2_L, F_Y_H2O_L,
                     F_U1, F_U2, F_U3, SPECIES_SLICES, BULK_SPECIES)
from pinn_model import PINNModel, PhysicsLoss, InletBCLoss, udf_inlet_temperature
from data_loader import (
    load_cas, build_multi_case_training_data, split_case_data, load_inlet_coords,
    make_collocation_points, make_data_points, make_initial_points,
    make_solid_temp_points, make_fluid_i_soild_temp_points,
    make_solid_temp_snapshot, make_fluid_i_soild_temp_snapshot, make_bc_points,
)

# Field name groups for logging
VEL_NAMES_P1 = ["u1", "v1", "w1"]
VEL_NAMES_P2 = ["u2", "v2", "w2"]
VEL_NAMES_P3 = ["u3", "v3", "w3"]
SCALAR_NAMES = ["p", "T"]
TURB_NAMES = ["k", "omega"]
VOF_NAMES = ["vof2", "vof3"]
SPECIES_P2_NAMES = ["Y_h2o2_v", "Y_h2o_v", "Y_air_v"]
SPECIES_P3_NAMES = ["Y_h2o2_l", "Y_h2o_l"]
ALL_FIELD_NAMES = (VEL_NAMES_P1 + VEL_NAMES_P2 + VEL_NAMES_P3 +
                    SCALAR_NAMES + TURB_NAMES + VOF_NAMES +
                    SPECIES_P2_NAMES + SPECIES_P3_NAMES)

PDE_LOG_NAMES = [
    "continuity",
    "momentum_p1_x", "momentum_p1_y", "momentum_p1_z",
    "momentum_p2_x", "momentum_p2_y", "momentum_p2_z",
    "momentum_p3_x", "momentum_p3_y", "momentum_p3_z",
    "energy_p2", "energy_p3", "energy_solid",
    "k_transport", "omega_transport",
    "species_h2o2_v", "species_h2o_v", "species_h2o2_l",
    "vof_advection",
]


class ParametricTrainer:
    def __init__(
        self,
        model: PINNModel,
        physics: PhysicsLoss,
        inlet_bc: InletBCLoss,
        optimizer: torch.optim.Optimizer,
        device: str,
        cfg: PINNConfig,
    ):
        self.model = model.to(device)
        self.physics = physics
        self.inlet_bc = inlet_bc
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.history = []
        self.globalStep = 0

    def get_pde_weight(self, name: str, fallback: float) -> float:
        return self.cfg.physics_loss_weights.get(name, fallback)

    def save_solid_temp_visualization(
        self, epoch: int,
        solid_viz_data: tuple | None,
        T_min: float, T_range: float,
    ) -> None:
        if solid_viz_data is None or self.cfg.solid_viz_interval <= 0:
            return
        if (epoch + 1) % self.cfg.solid_viz_interval != 0:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ModuleNotFoundError:
            return
        _ = Axes3D
        x_solid, t_solid, T_solid, _, bc_solid = solid_viz_data
        if len(x_solid) == 0:
            return
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_solid.to(self.device), t_solid.to(self.device), bc_solid.to(self.device))
        coord = x_solid.cpu().numpy()
        T_true = (T_solid.cpu().numpy().reshape(-1) * T_range) + T_min
        T_pred = (pred[:, F_T].cpu().numpy().reshape(-1) * T_range) + T_min
        T_error = T_pred - T_true
        rmse = float(np.sqrt(np.mean(T_error ** 2)))
        mae = float(np.mean(np.abs(T_error)))
        viz_dir = self.cfg.output_dir / "solid_temp_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(13, 11))
        axes = [
            fig.add_subplot(2, 2, 1, projection="3d"),
            fig.add_subplot(2, 2, 2, projection="3d"),
            fig.add_subplot(2, 2, 3, projection="3d"),
            fig.add_subplot(2, 2, 4),
        ]
        for ax, vals, title, cmap in [
            (axes[0], T_true, "fluid_i-soild — actual T (K)", "viridis"),
            (axes[1], T_pred, "fluid_i-soild — predicted T (K)", "viridis"),
            (axes[2], T_error, f"fluid_i-soild error (K), RMSE={rmse:.2f}, MAE={mae:.2f}", "coolwarm"),
        ]:
            contour = ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c=vals, s=2, cmap=cmap, alpha=0.85)
            ax.set_title(title); ax.set_xlabel("x_norm"); ax.set_ylabel("y_norm"); ax.set_zlabel("z_norm")
            ax.view_init(elev=22, azim=-55)
            fig.colorbar(contour, ax=ax)
        ax = axes[3]
        ax.scatter(T_true, T_pred, s=8, alpha=0.5)
        min_v, max_v = min(T_true.min(), T_pred.min()), max(T_true.max(), T_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
        ax.set_title("fluid_i-soild: Predicted vs actual T")
        ax.set_xlabel("Actual T (K)"); ax.set_ylabel("Predicted T (K)")
        fig.suptitle(f"Inner Surface T — Epoch {epoch+1} ({len(coord):,} cells, RMSE={rmse:.2f}K, MAE={mae:.2f}K)",
                     fontsize=14)
        fig.tight_layout()
        out_path = viz_dir / f"epoch_{epoch + 1:05d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def evaluate_validation(self, val_data_loader, val_solid_loader=None) -> dict:
        metrics = defaultdict(list)
        self.model.eval()
        with torch.no_grad():
            for x_dat, t_dat, v_dat, bc_dat in val_data_loader:
                pred = self.model(x_dat.to(self.device), t_dat.to(self.device), bc_dat.to(self.device))
                target = v_dat.to(self.device)
                field_mse = ((pred - target) ** 2).mean(dim=0)
                metrics["val_loss_data"].append(field_mse.mean().item())
                for i, name in enumerate(ALL_FIELD_NAMES):
                    metrics[f"val_data_{name}"].append(field_mse[i].item())
            if val_solid_loader is not None:
                for x_solid, t_solid, T_solid, _, bc_solid in val_solid_loader:
                    pred = self.model(x_solid.to(self.device), t_solid.to(self.device), bc_solid.to(self.device))
                    target = T_solid.to(self.device)
                    metrics["val_loss_solid_temp"].append(((pred[:, F_T:F_T + 1] - target) ** 2).mean().item())
        return {key: float(np.mean(value)) for key, value in metrics.items() if value}

    def train_step(
        self,
        x_col, t_col, bc_col,
        x_dat, t_dat, v_dat, bc_dat,
        t_bc, bc_bc, T_bc_a, T_bc_b,
        x_init, v_init, bc_init,
        x_solid, t_solid, T_solid_target, grad_T_solid, bc_solid,
        solid_mask,
        lambda_physics: float = 0.1,
        lambda_bc: float = 10.0,
        lambda_data: float = 1.0,
    ) -> tuple[float, dict]:
        self.optimizer.zero_grad()
        loss = 0.0
        metrics = {}

        # ── 1. Physics PDE loss ──
        if x_col is not None:
            xc = x_col.requires_grad_(True)
            tc = t_col.requires_grad_(True)
            pred = self.model(xc, tc, bc_col)
            pde_res = self.physics.compute(xc, tc, pred, solid_mask=None)

            if solid_mask is not None:
                fluid_mask = ~solid_mask
                solid_m = solid_mask
            else:
                fluid_mask = torch.ones(len(xc), dtype=torch.bool, device=xc.device)
                solid_m = torch.zeros(len(xc), dtype=torch.bool, device=xc.device)

            physics_loss = 0.0
            solid_pdes = {"energy_solid"}
            for name, res in pde_res.items():
                if torch.isnan(res).any():
                    metrics[f'nan_{name}'] = 1
                    continue
                if name in solid_pdes:
                    if solid_m.sum() > 0:
                        loss_item = (res[solid_m] ** 2).mean()
                        w = self.get_pde_weight(name, self.cfg.lambda_physics_solid)
                else:
                    if fluid_mask.sum() > 0:
                        loss_item = (res[fluid_mask] ** 2).mean()
                        w = self.get_pde_weight(name, lambda_physics)
                if (name in solid_pdes and solid_m.sum() > 0) or (name not in solid_pdes and fluid_mask.sum() > 0):
                    physics_loss += w * loss_item
                    metrics[f'loss_{name}'] = loss_item.item()
                    metrics[f'weighted_loss_{name}'] = (w * loss_item).item()

            loss += physics_loss
            metrics['loss_physics'] = float(physics_loss.item())

        # ── 2. Data fitting (all 20 fields) ──
        # Reuse this pred for species/VOF constraints below (avoid redundant forward passes)
        pred_dat = None
        if x_dat is not None:
            pred_dat = self.model(x_dat, t_dat, bc_dat)
            data_diff = pred_dat - v_dat
            field_mse = (data_diff ** 2).mean(dim=0)
            d_loss = field_mse.mean()
            loss += lambda_data * d_loss
            metrics['loss_data'] = d_loss.item()
            for i, name in enumerate(ALL_FIELD_NAMES):
                metrics[f'loss_data_{name}'] = field_mse[i].item()

        # ── 3. Solid temperature supervision ──
        if x_solid is not None and T_solid_target is not None:
            pred_solid = self.model(x_solid, t_solid, bc_solid)
            T_pred = pred_solid[:, F_T:F_T + 1]
            solid_T_loss = ((T_pred - T_solid_target) ** 2).mean()
            loss += self.cfg.lambda_solid_temp * solid_T_loss
            metrics['loss_solid_temp'] = solid_T_loss.item()

            # ── 3b. Solid temperature spatial gradient supervision ──
            if grad_T_solid is not None and grad_T_solid.abs().sum() > 0:
                x_s = x_solid.detach().clone().requires_grad_(True)
                pred_s = self.model(x_s, t_solid, bc_solid)
                T_s = pred_s[:, F_T:F_T + 1]
                dT = grad(T_s, x_s, torch.ones_like(T_s),
                          create_graph=True, retain_graph=True)[0]
                # dT: [N, 3] — predicted gradient in normalized coords
                # grad_T_solid: [N, 3] — CFD gradient in normalized coords
                grad_loss = ((dT - grad_T_solid) ** 2).mean()
                loss += self.cfg.lambda_solid_grad * grad_loss
                metrics['loss_solid_grad'] = grad_loss.item()

        # ── 4. Inlet BC loss ──
        if t_bc is not None and self.globalStep % self.cfg.bc_loss_interval == 0:
            bc_loss, bc_metrics = self.inlet_bc(self.model, t_bc, bc_bc, T_bc_a, T_bc_b)
            loss += lambda_bc * bc_loss
            metrics.update(bc_metrics)

        # ── 5. Initial condition loss ──
        if x_init is not None:
            t0 = torch.zeros(x_init.shape[0], 1, device=self.device)
            pred_init = self.model(x_init, t0, bc_init)
            i_loss = ((pred_init - v_init) ** 2).mean()
            loss += self.cfg.lambda_initial * i_loss
            metrics['loss_initial'] = i_loss.item()

        # ── 6. Species sum constraint (reuses pred_dat from step 2) ──
        if pred_dat is not None:
            sp2_sum = pred_dat[:, F_Y_H2O2_V] + pred_dat[:, F_Y_H2O_V] + pred_dat[:, F_Y_AIR_V]
            sp2_loss = ((sp2_sum - 1.0) ** 2).mean()
            sp3_sum = pred_dat[:, F_Y_H2O2_L] + pred_dat[:, F_Y_H2O_L]
            sp3_loss = ((sp3_sum - 1.0) ** 2).mean()
            sp_loss = sp2_loss + sp3_loss
            loss += self.cfg.lambda_species * sp_loss
            metrics['loss_species_sum'] = sp_loss.item()

            # ── 7. VOF sum constraint (reuses pred_dat from step 2) ──
            vof_sum = pred_dat[:, F_VOF2] + pred_dat[:, F_VOF3]
            vof_excess = torch.clamp(vof_sum - 1.0, min=0.0)
            vof_loss = (vof_excess ** 2).mean()
            loss += self.cfg.lambda_vof * vof_loss
            metrics['loss_vof_constraint'] = vof_loss.item()

        if torch.isnan(loss):
            print("WARNING: NaN loss detected!")
            return float('inf'), metrics

        loss.backward()
        self.optimizer.step()
        self.globalStep += 1
        metrics['loss_total'] = loss.item()
        return loss.item(), metrics

    def fit(
        self,
        colloc_loader, data_loader, bc_loader, initial_loader,
        solid_temp_loader=None, val_data_loader=None, val_solid_loader=None,
        n_epochs: int = 50000, patience: int = 3000, log_every: int = 100,
        training_data: dict | None = None, rng=None,
        solid_viz_data=None, T_min: float = 0.0, T_range: float = 1.0,
    ) -> list:
        best = float("inf")
        no_improve = 0

        all_colloc = list(colloc_loader) if colloc_loader else []
        all_data = list(data_loader) if data_loader else []
        dynamic_bc = training_data is not None and rng is not None
        all_bc = [] if dynamic_bc else list(bc_loader) if bc_loader else []
        all_init = list(initial_loader) if initial_loader else []
        all_solid = list(solid_temp_loader) if solid_temp_loader else []

        n_col = len(all_colloc); n_dat = len(all_data)
        n_init = len(all_init); n_solid = len(all_solid)
        max_batches = max(n_col or 1, n_dat or 1, n_init or 1, n_solid or 1)

        if dynamic_bc:
            print(f"  Batches per epoch: colloc={n_col}, data={n_dat}, bc=dynamic, init={n_init}, solid_T={n_solid}")
        else:
            print(f"  Batches per epoch: colloc={n_col}, data={n_dat}, bc={len(all_bc)}, init={n_init}, solid_T={n_solid}")

        for epoch in range(n_epochs):
            self.model.train()
            epoch_losses = []
            epoch_metrics = defaultdict(list)

            for i in range(max_batches):
                if n_col > 0:
                    x_col, t_col, mask_col, bc_col = all_colloc[i % n_col]
                else:
                    x_col = t_col = mask_col = bc_col = None

                x_dat, t_dat, v_dat, bc_dat = all_data[i % n_dat] if n_dat > 0 else (None, None, None, None)

                if dynamic_bc:
                    t_bc, bc_bc, T_bc_a, T_bc_b = make_bc_points(training_data, self.cfg.batch_size_boundary, rng)
                else:
                    t_bc = bc_bc = T_bc_a = T_bc_b = None

                x_init, v_init, bc_init = all_init[i % n_init] if n_init > 0 else (None, None, None)
                x_solid, t_solid, T_solid, grad_T_solid, bc_solid = all_solid[i % n_solid] if n_solid > 0 else (None, None, None, None, None)

                loss_val, metrics = self.train_step(
                    x_col.to(self.device) if x_col is not None else None,
                    t_col.to(self.device) if t_col is not None else None,
                    bc_col.to(self.device) if bc_col is not None else None,
                    x_dat.to(self.device) if x_dat is not None else None,
                    t_dat.to(self.device) if t_dat is not None else None,
                    v_dat.to(self.device) if v_dat is not None else None,
                    bc_dat.to(self.device) if bc_dat is not None else None,
                    t_bc.to(self.device) if t_bc is not None else None,
                    bc_bc.to(self.device) if bc_bc is not None else None,
                    T_bc_a.to(self.device) if T_bc_a is not None else None,
                    T_bc_b.to(self.device) if T_bc_b is not None else None,
                    x_init.to(self.device) if x_init is not None else None,
                    v_init.to(self.device) if v_init is not None else None,
                    bc_init.to(self.device) if bc_init is not None else None,
                    x_solid.to(self.device) if x_solid is not None else None,
                    t_solid.to(self.device) if t_solid is not None else None,
                    T_solid.to(self.device) if T_solid is not None else None,
                    grad_T_solid.to(self.device) if grad_T_solid is not None else None,
                    bc_solid.to(self.device) if bc_solid is not None else None,
                    mask_col.to(self.device) if mask_col is not None else None,
                    lambda_physics=self.cfg.lambda_physics_fluid,
                    lambda_bc=self.cfg.lambda_bc_inlet,
                    lambda_data=self.cfg.lambda_data,
                )
                epoch_losses.append(loss_val)
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        epoch_metrics[k].append(v)

            avg_loss = np.mean(epoch_losses)
            self.history.append(avg_loss)

            if epoch % log_every == 0 or epoch == n_epochs - 1:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}/{n_epochs}")
                print(f"{'='*60}")
                print(f"  Weighted Total Loss: {avg_loss:.6f}  (best={best:.6f})")

                if 'loss_physics' in epoch_metrics:
                    print(f"\n  [Physics Residual MSE]")
                    print(f"    Weighted Total: {np.mean(epoch_metrics['loss_physics']):.6f}")
                    for name in PDE_LOG_NAMES:
                        key = f'loss_{name}'
                        if key in epoch_metrics and len(epoch_metrics[key]) > 0:
                            val = np.mean(epoch_metrics[key])
                            wkey = f'weighted_loss_{name}'
                            wval = np.mean(epoch_metrics[wkey]) if wkey in epoch_metrics else val
                            print(f"    {name:20s}: mse={val:.6e}, weighted={wval:.6e}")

                if 'loss_data' in epoch_metrics:
                    print(f"\n  [Data MSE]")
                    print(f"    Total: {np.mean(epoch_metrics['loss_data']):.6f}")
                    # Phase 1 velocity
                    p1_vals = [np.mean(epoch_metrics.get(f'loss_data_{n}', [0])) for n in VEL_NAMES_P1]
                    print(f"    P1 vel:  u1={p1_vals[0]:.6f} v1={p1_vals[1]:.6f} w1={p1_vals[2]:.6f}")
                    p2_vals = [np.mean(epoch_metrics.get(f'loss_data_{n}', [0])) for n in VEL_NAMES_P2]
                    print(f"    P2 vel:  u2={p2_vals[0]:.6f} v2={p2_vals[1]:.6f} w2={p2_vals[2]:.6f}")
                    p3_vals = [np.mean(epoch_metrics.get(f'loss_data_{n}', [0])) for n in VEL_NAMES_P3]
                    print(f"    P3 vel:  u3={p3_vals[0]:.6f} v3={p3_vals[1]:.6f} w3={p3_vals[2]:.6f}")
                    for grp_name, grp in [("Scalar", SCALAR_NAMES), ("Turb", TURB_NAMES),
                                          ("VOF", VOF_NAMES), ("Sp2", SPECIES_P2_NAMES), ("Sp3", SPECIES_P3_NAMES)]:
                        vals = [np.mean(epoch_metrics.get(f'loss_data_{n}', [0])) for n in grp]
                        print(f"    {grp_name:6s}: " + " ".join(f"{n}={v:.6f}" for n, v in zip(grp, vals)))

                if 'loss_solid_temp' in epoch_metrics:
                    print(f"    Solid T:  {np.mean(epoch_metrics['loss_solid_temp']):.6f}")
                if 'loss_solid_grad' in epoch_metrics:
                    print(f"    Solid Grad: {np.mean(epoch_metrics['loss_solid_grad']):.6e}")
                if 'loss_initial' in epoch_metrics:
                    print(f"    Initial:  {np.mean(epoch_metrics['loss_initial']):.6f}")

                if 'bc_inlet_a' in epoch_metrics:
                    print(f"\n  [Inlet BC]")
                    print(f"    Inlet A:  {np.mean(epoch_metrics['bc_inlet_a']):.6f}")
                    print(f"    Inlet B:  {np.mean(epoch_metrics['bc_inlet_b']):.6f}")

                if 'loss_species_sum' in epoch_metrics:
                    print(f"    Species sum constraint: {np.mean(epoch_metrics['loss_species_sum']):.6f}")
                if 'loss_vof_constraint' in epoch_metrics:
                    print(f"    VOF constraint:         {np.mean(epoch_metrics['loss_vof_constraint']):.6f}")

                if val_data_loader is not None:
                    val_metrics = self.evaluate_validation(val_data_loader, val_solid_loader)
                    if val_metrics:
                        print(f"\n  [Validation MSE]")
                        if 'val_loss_data' in val_metrics:
                            print(f"    Data:     {val_metrics['val_loss_data']:.6f}")
                        if 'val_loss_solid_temp' in val_metrics:
                            print(f"    Solid T:  {val_metrics['val_loss_solid_temp']:.6f}")

                print(f"\n  Patience: {no_improve}/{patience}")

            if avg_loss < best:
                best = avg_loss
                no_improve = 0
                print(f"  -> New best! Saved to model_best.pt")
                torch.save(self.model.state_dict(), self.cfg.output_dir / "model_best.pt")
            else:
                no_improve += 1

            if patience > 0 and no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            self.save_solid_temp_visualization(epoch, solid_viz_data, T_min, T_range)

        return self.history


def main():
    print("=" * 60)
    print("PINN for H2O2 Sterilization — Multi-Fluid 3-Phase Model")
    print("=" * 60)

    data_cfg = DataConfig()
    pinn_cfg = PINNConfig()
    rng = np.random.default_rng(42)

    device = torch.device(pinn_cfg.device)
    print(f"Device: {device}")
    print(f"Output fields: {N_OUTPUT_FIELDS}  ({', '.join(ALL_FIELD_NAMES)})")

    # [1] Inlet coords
    print("\n[1/7] Loading inlet coordinates...")
    inlet_a_coords, inlet_b_coords = load_inlet_coords(data_cfg.inlet_coords_path)
    print(f"  inlet_a: {inlet_a_coords.shape[0]} faces, inlet_b: {inlet_b_coords.shape[0]} faces")

    # [2] Mesh + data
    print("\n[2/7] Loading mesh and multi-case multi-fluid training data...")
    mesh = load_cas(data_cfg.cas_path)
    data = build_multi_case_training_data(mesh, data_cfg.data_root, data_cfg.dat_glob,
                                           data_cfg.case_glob,
                                           use_discrete_time_input=pinn_cfg.use_discrete_time_input)
    print(f"  n_cases: {data['n_cases']}")
    print(f"  n_fluid: {data['n_fluid']}  n_solid: {data['n_solid']}")
    print(f"  coord_range: {data['coord_range']}")
    print(f"  time_range:  [{data['time_min']:.2f}, {data['time_max']:.2f}]s")
    print(f"  T range:     [{data['T_min']:.2f}, {data['T_range'] + data['T_min']:.2f}]K")

    train_data, val_data = split_case_data(data, pinn_cfg.val_case_ratio, rng)
    train_names = train_data.get('case_names', [])
    val_names = val_data.get('case_names', []) if val_data else []
    print(f"  train_cases ({train_data['n_cases']}): {train_names}")
    if val_data:
        print(f"  val_cases ({val_data['n_cases']}): {val_names}")

    # Save train/val split info to output dir
    pinn_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = pinn_cfg.output_dir / "train_val_split.txt"
    with open(split_path, "w", encoding="utf-8") as f:
        f.write(f"Total cases: {data['n_cases']}\n")
        f.write(f"val_case_ratio: {pinn_cfg.val_case_ratio}\n")
        f.write(f"Random seed: 42\n\n")
        f.write(f"=== Training cases ({len(train_names)}) ===\n")
        for name in train_names:
            f.write(f"  {name}\n")
        f.write(f"\n=== Validation cases ({len(val_names)}) ===\n")
        for name in val_names:
            f.write(f"  {name}\n")
        f.write(f"\nNote: Validation = cross-case generalization test.\n")
        f.write(f"  The model is trained on the training case(s) and\n")
        f.write(f"  validated on the held-out case(s) with different\n")
        f.write(f"  operating conditions (T_preheat, T_h2o2).\n")
    print(f"  Split info saved to: {split_path}")

    # [3] Model
    print("\n[3/7] Creating multi-fluid PINN model...")
    model = PINNModel(
        n_spatial=3, hidden=pinn_cfg.hidden_size, n_layers=pinn_cfg.n_hidden_layers,
        activation=pinn_cfg.activation, n_output=N_OUTPUT_FIELDS,
        parametric_bc=pinn_cfg.parametric_bc,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Hidden: {pinn_cfg.hidden_size}x{pinn_cfg.n_hidden_layers},  Params: {n_params:,}")

    # [4] Physics
    print("\n[4/7] Creating multi-phase physics loss...")
    physics = PhysicsLoss(
        rho_p1=pinn_cfg.rho_p1, mu_p1=pinn_cfg.mu_p1, cp_p1=pinn_cfg.cp_p1, k_p1=pinn_cfg.k_p1,
        rho_p2=pinn_cfg.rho_p2, mu_p2=pinn_cfg.mu_p2, cp_p2=pinn_cfg.cp_p2, k_p2=pinn_cfg.k_p2,
        rho_p3=pinn_cfg.rho_p3, mu_p3=pinn_cfg.mu_p3, cp_p3=pinn_cfg.cp_p3, k_p3=pinn_cfg.k_p3,
        rho_solid=pinn_cfg.rho_solid, cp_solid=pinn_cfg.cp_solid, k_solid=pinn_cfg.k_solid,
        D_h2o2_v=pinn_cfg.D_h2o2_v, D_h2o_v=pinn_cfg.D_h2o_v,
        D_h2o2_l=pinn_cfg.D_h2o2_l,
        K_drag_12=pinn_cfg.K_drag_12, K_drag_13=pinn_cfg.K_drag_13,
        enabled=pinn_cfg.enabled_pdes,
        coord_scale=torch.tensor(data["coord_range"], dtype=torch.float32),
        time_scale=float(data["time_scale"]),
        field_mean=torch.tensor(
            np.array([data["scales"][i].get("mean", 0.0) for i in range(N_OUTPUT_FIELDS)]),
            dtype=torch.float32,
        ),
        field_std=torch.tensor(
            np.array([data["scales"][i].get("std", 1.0) for i in range(N_OUTPUT_FIELDS)]),
            dtype=torch.float32,
        ),
    )
    print(f"  Enabled PDEs: {len(pinn_cfg.enabled_pdes)}")

    # [5] Inlet BC
    print("\n[5/7] Creating inlet BC loss...")
    inlet_bc = InletBCLoss(
        inlet_a_coords, inlet_b_coords,
        coord_min=data["coord_min"], coord_max=data["coord_max"],
        time_min=data["time_min"], time_max=data["time_max"],
        T_min=data["T_min"], T_max=data["T_range"] + data["T_min"],
        device=str(device), hard_inlet_bc=pinn_cfg.hard_inlet_bc,
    )

    # [6] Data loaders
    print("\n[6/7] Creating data loaders...")
    # Collocation
    n_col = pinn_cfg.n_collocation_points
    x_col, t_col, mask_col, bc_col = make_collocation_points(train_data, n_col, rng, perturb_sigma=0.01)
    colloc_loader = DataLoader(TensorDataset(x_col, t_col, mask_col, bc_col),
                               batch_size=pinn_cfg.batch_size, shuffle=True)
    print(f"  collocation: {n_col} points, solid ratio={mask_col.float().mean():.3f}")

    # Data
    n_dat = pinn_cfg.n_data_points
    x_dat, t_dat, v_dat, bc_dat = make_data_points(train_data, n_dat, rng)
    data_loader_ = DataLoader(TensorDataset(x_dat, t_dat, v_dat, bc_dat),
                              batch_size=pinn_cfg.batch_size, shuffle=True)
    print(f"  data:        {n_dat} points, {N_OUTPUT_FIELDS} fields each")

    # BC (dynamic)
    bc_loader = None
    print(f"  BC:          dynamic, {pinn_cfg.batch_size_boundary} samples/step")

    # Initial
    n_init = pinn_cfg.n_initial_points
    x_init, v_init, bc_init = make_initial_points(train_data, n_init, rng)
    initial_loader = DataLoader(TensorDataset(x_init, v_init, bc_init),
                                batch_size=pinn_cfg.batch_size, shuffle=True)
    print(f"  initial:     {n_init} points")

    # Solid T — focused on fluid_i-soild surface (sterilization-critical region)
    n_solid_pts = pinn_cfg.n_solid_temp_points
    x_s, t_s, T_s, grad_T_s, bc_s = make_fluid_i_soild_temp_points(train_data, n_solid_pts, rng)
    if len(x_s) > 0:
        solid_temp_loader = DataLoader(TensorDataset(x_s, t_s, T_s, grad_T_s, bc_s),
                                       batch_size=pinn_cfg.batch_size, shuffle=True)
        solid_viz_data = make_fluid_i_soild_temp_snapshot(train_data, pinn_cfg.solid_viz_case_idx, pinn_cfg.solid_viz_time_idx)
        print(f"  solid_T:     {n_solid_pts} points (inner-surface focused), "
              f"viz every {pinn_cfg.solid_viz_interval} epochs")
    else:
        solid_temp_loader = None; solid_viz_data = None
        print(f"  solid_T:     SKIP")

    # Validation
    val_data_loader = val_solid_loader = None
    if val_data is not None:
        x_val, t_val, v_val, bc_val = make_data_points(val_data, pinn_cfg.n_val_data_points, rng)
        val_data_loader = DataLoader(TensorDataset(x_val, t_val, v_val, bc_val),
                                     batch_size=pinn_cfg.batch_size, shuffle=False)
        x_vs, t_vs, T_vs, grad_T_vs, bc_vs = make_solid_temp_points(val_data, pinn_cfg.n_val_solid_temp_points, rng)
        if len(x_vs) > 0:
            val_solid_loader = DataLoader(TensorDataset(x_vs, t_vs, T_vs, grad_T_vs, bc_vs),
                                          batch_size=pinn_cfg.batch_size, shuffle=False)
        print(f"  validation:  {pinn_cfg.n_val_data_points} fluid + {len(x_vs)} solid points")

    # [7] Train
    optimizer = torch.optim.Adam(model.parameters(), lr=pinn_cfg.learning_rate)
    print("\n[7/7] Training...")
    print(f"  Epochs: {pinn_cfg.n_epochs}, LR: {pinn_cfg.learning_rate:.2e}, Patience: {pinn_cfg.patience}")
    pinn_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("  Case temperatures:")
    for case in train_data["cases"]:
        print(f"    {case['case_name']}: T_preheat={case['T_preheat_K']-273.15:.0f}°C, "
              f"T_h2o2={case['T_h2o2_K']-273.15:.0f}°C")

    trainer = ParametricTrainer(model, physics, inlet_bc, optimizer, str(device), pinn_cfg)

    history = trainer.fit(
        colloc_loader, data_loader_, bc_loader, initial_loader, solid_temp_loader,
        val_data_loader=val_data_loader, val_solid_loader=val_solid_loader,
        n_epochs=pinn_cfg.n_epochs, patience=pinn_cfg.patience, log_every=10,
        training_data=train_data, rng=rng,
        solid_viz_data=solid_viz_data,
        T_min=data["T_min"], T_range=data["T_range"],
    )

    torch.save(model.state_dict(), pinn_cfg.output_dir / "model.pt")
    np.save(pinn_cfg.output_dir / "train_loss.npy", np.array(history))
    print(f"\nSaved to {pinn_cfg.output_dir}")
    print(f"Final loss: {history[-1]:.6f},  Best loss: {min(history):.6f}")


if __name__ == "__main__":
    main()
