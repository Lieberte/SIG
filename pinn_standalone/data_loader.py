"""
Data loader for multi-fluid PINN. Reads cas.h5 mesh + dat.h5 fields.

3-phase Eulerian-Eulerian model:
  Phase 1 — carrier gas (air)  : u,v,w,p,k,ω   (610009 fluid cells)
  Phase 2 — vapor               : u,v,w,T,α,3×Y  (610009 fluid cells)
  Phase 3 — liquid              : u,v,w,T,α,2×Y  (610009 fluid cells)
  Solid                         : T              (64503 cells)

20-field output per fluid cell (see config.py for layout).
"""
import h5py
import numpy as np
from pathlib import Path
from config import (
    N_OUTPUT_FIELDS, FIELD_NAMES,
    F_U1, F_V1, F_W1, F_U2, F_V2, F_W2, F_U3, F_V3, F_W3,
    F_P, F_T, F_K, F_OMEGA, F_VOF2, F_VOF3,
    F_Y_H2O2_V, F_Y_H2O_V, F_Y_AIR_V,
    F_Y_H2O2_L, F_Y_H2O_L,
    VEL_SLICES, SPECIES_SLICES, BULK_SPECIES,
    SV_Y_TO_FIELD_P2, SV_Y_TO_FIELD_P3,
)


def parse_names(raw: np.ndarray) -> list[str]:
    arr = np.asarray(raw).reshape(-1)
    if arr.size == 0:
        return []
    first = arr[0]
    text = first.decode("utf-8", errors="ignore") if isinstance(first, bytes) else str(first)
    return [p.strip() for p in text.split(";") if p.strip()]


def load_cas(cas_path: Path) -> dict:
    """Extract mesh topology from cas.h5."""
    with h5py.File(cas_path, "r") as f:
        node_coords = np.asarray(f["meshes/1/nodes/coords/1"][()])
        face_nn = np.asarray(f["meshes/1/faces/nodes/1/nnodes"][()])
        face_nodes = np.asarray(f["meshes/1/faces/nodes/1/nodes"][()])
        face_c0 = np.asarray(f["meshes/1/faces/c0/1"][()])
        offsets = np.cumsum(np.concatenate([[0], face_nn]))

        z = f["meshes/1/faces/zoneTopology"]
        zone_ids = np.asarray(z["id"][()]).reshape(-1)
        zone_mins = np.asarray(z["minId"][()]).reshape(-1)
        zone_maxs = np.asarray(z["maxId"][()]).reshape(-1)
        zone_c1 = np.asarray(z["c1"][()]).reshape(-1)
        zone_names = parse_names(np.asarray(z["name"][()]))

    zones = {}
    for i in range(len(zone_ids)):
        n = int(zone_maxs[i] - zone_mins[i] + 1)
        zones[zone_names[i] if i < len(zone_names) else f"zone_{zone_ids[i]}"] = {
            "id": int(zone_ids[i]),
            "face_min": int(zone_mins[i]),
            "face_max": int(zone_maxs[i]),
            "n_faces": n,
            "is_boundary": int(zone_c1[i]) == 0,
        }

    def compute_face_centers(fid_range):
        f0, f1 = fid_range
        n = f1 - f0 + 1
        centers = np.zeros((n, 3))
        for i in range(n):
            fid = f0 - 1 + i
            nids = face_nodes[offsets[fid]:offsets[fid + 1]]
            centers[i] = node_coords[nids - 1].mean(axis=0)
        return centers

    return {
        "node_coords": node_coords,
        "face_nodes": face_nodes,
        "face_nn": face_nn,
        "offsets": offsets,
        "face_c0": face_c0,
        "zones": zones,
        "n_nodes": node_coords.shape[0],
        "n_faces": len(face_nn),
        "compute_face_centers": compute_face_centers,
    }


def compute_cell_centers(mesh: dict) -> np.ndarray:
    """Compute cell center coordinates from node coords and face-nodes."""
    node_coords = mesh["node_coords"]
    face_nodes = mesh["face_nodes"]
    offsets = mesh["offsets"]
    face_c0 = mesh["face_c0"]
    n_faces = mesh["n_faces"]

    n_cells = int(face_c0.max()) + 1
    cell_node_sets = [set() for _ in range(n_cells)]

    for fid in range(n_faces):
        owner = int(face_c0[fid])
        if owner < 0 or owner >= n_cells:
            continue
        for nid in face_nodes[offsets[fid]:offsets[fid + 1]]:
            cell_node_sets[owner].add(int(nid) - 1)

    cell_centers = np.zeros((n_cells, 3), dtype=np.float64)
    for cid in range(n_cells):
        nids = np.array(list(cell_node_sets[cid]))
        if len(nids) == 0:
            continue
        cell_centers[cid] = node_coords[nids].mean(axis=0)

    return cell_centers


# ── multi-fluid field loading ─────────────────────────────────

def _safe_read(dataset_path) -> np.ndarray | None:
    """Read a dataset if it exists, else return None."""
    try:
        return np.asarray(dataset_path[()])
    except (KeyError, TypeError):
        return None


def load_multi_fluid_fields(dat_dir: Path, dat_glob: str) -> tuple:
    """Load cell-centered fields for all 3 phases from all dat.h5 files.

    Returns:
        times: (n_time,)  physical times
        fluid_data: (n_time, n_fluid, N_OUTPUT_FIELDS)  full 20-field array
        solid_temp: (n_time, n_solid)  solid temperature only
        phase_sizes: dict with n_fluid, n_solid
    """
    dat_files = sorted(dat_dir.glob(dat_glob))
    if not dat_files:
        raise FileNotFoundError(f"No dat files in {dat_dir} matching {dat_glob}")

    times = []
    fluid_data_list = []   # each entry: (n_fluid, 20)
    solid_temp_list = []   # each entry: (n_solid,)

    for dat_file in dat_files:
        t_str = dat_file.stem.split("-")[-1].replace(".dat", "")
        times.append(float(t_str))

        with h5py.File(dat_file, "r") as f:
            # ── Phase 1: carrier gas (velocity, pressure, turbulence) ──
            p1 = "results/1/phase-1/cells"
            sv_u1  = np.asarray(f[f"{p1}/SV_U/1"][()])
            sv_v1  = np.asarray(f[f"{p1}/SV_V/1"][()])
            sv_w1  = np.asarray(f[f"{p1}/SV_W/1"][()])
            sv_p   = np.asarray(f[f"{p1}/SV_P/1"][()])
            sv_k   = np.asarray(f[f"{p1}/SV_K/1"][()])
            sv_o   = np.asarray(f[f"{p1}/SV_O/1"][()])

            n_fluid = len(sv_u1)

            # ── Phase 2: vapor (velocity, temperature, VOF, 3 species) ──
            p2 = "results/1/phase-2/cells"
            sv_u2  = np.asarray(f[f"{p2}/SV_U/1"][()])
            sv_v2  = np.asarray(f[f"{p2}/SV_V/1"][()])
            sv_w2  = np.asarray(f[f"{p2}/SV_W/1"][()])
            sv_t2  = np.asarray(f[f"{p2}/SV_T/1"][()])
            sv_vof2 = np.asarray(f[f"{p2}/SV_VOF/1"][()])
            sv_y2  = np.asarray(f[f"{p2}/SV_Y/1"][()])  # (n_fluid, 3)

            # ── Phase 3: liquid (velocity, temperature, VOF, 2 species) ──
            p3 = "results/1/phase-3/cells"
            sv_u3  = np.asarray(f[f"{p3}/SV_U/1"][()])
            sv_v3  = np.asarray(f[f"{p3}/SV_V/1"][()])
            sv_w3  = np.asarray(f[f"{p3}/SV_W/1"][()])
            sv_t3  = np.asarray(f[f"{p3}/SV_T/1"][()])
            sv_vof3 = np.asarray(f[f"{p3}/SV_VOF/1"][()])
            sv_y3  = np.asarray(f[f"{p3}/SV_Y/1"][()])  # (n_fluid, 2)

            # ── Solid: temperature ──
            sv_t_solid = _safe_read(f["results/1/phase-1/cells/SV_T/1"])
            if sv_t_solid is None:
                sv_t_solid = np.zeros(0)
            n_solid = len(sv_t_solid)

            # ── Assemble 20-field array ──
            n_fluid_actual = len(sv_u1)
            fd = np.zeros((n_fluid_actual, N_OUTPUT_FIELDS), dtype=np.float32)

            fd[:, F_U1] = sv_u1; fd[:, F_V1] = sv_v1; fd[:, F_W1] = sv_w1
            fd[:, F_U2] = sv_u2; fd[:, F_V2] = sv_v2; fd[:, F_W2] = sv_w2
            fd[:, F_U3] = sv_u3; fd[:, F_V3] = sv_v3; fd[:, F_W3] = sv_w3
            fd[:, F_P]  = sv_p
            fd[:, F_K]  = sv_k;  fd[:, F_OMEGA] = sv_o
            fd[:, F_VOF2] = sv_vof2; fd[:, F_VOF3] = sv_vof3

            # Temperature: average phase 2 and 3 (thermal equilibrium assumption)
            fd[:, F_T] = (sv_t2 + sv_t3) * 0.5

            # Phase 2 species — use config-defined SV_Y column mapping
            for sv_y_col, field_idx in SV_Y_TO_FIELD_P2.items():
                fd[:, field_idx] = sv_y2[:, sv_y_col]

            # Phase 3 species — use config-defined SV_Y column mapping
            for sv_y_col, field_idx in SV_Y_TO_FIELD_P3.items():
                fd[:, field_idx] = sv_y3[:, sv_y_col]

            fluid_data_list.append(fd)
            solid_temp_list.append(sv_t_solid.astype(np.float32))

    return (
        np.array(times, dtype=np.float64),
        fluid_data_list,
        solid_temp_list,
        {"n_fluid": n_fluid, "n_solid": n_solid},
    )


def parse_temperatures_from_path(dat_dir: Path) -> tuple[float, float]:
    """Parse T_preheat and T_h2o2 from folder name (e.g. T_160_200 → 433.15, 473.15 K)."""
    import re
    name = dat_dir.name
    match = re.search(r'T_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)', name, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 273.15, float(match.group(2)) + 273.15
    match2 = re.search(r'Tpre(\d+(?:\.\d+)?).*?Th2o2(\d+(?:\.\d+)?)', name, re.IGNORECASE)
    if match2:
        return float(match2.group(1)) + 273.15, float(match2.group(2)) + 273.15
    raise ValueError(f"Cannot parse temperatures from folder name '{name}'")


# ── Normalization ─────────────────────────────────────────────

def _compute_scales(all_fluid: np.ndarray) -> dict:
    """Compute per-field normalization parameters from stacked fluid data.

    Strategy:
      - Velocity fields (9 total): zero-mean, unit-variance-per-phase
      - Pressure: zero-mean, std scaling
      - Temperature: min-max globally
      - Turbulence (k,ω): log-transform then min-max
      - VOF: [0,1] no normalization
      - Species: [0,1] no normalization (already mass fractions)
    """
    scales = {}
    # Velocity: per-component standardization  (zero mean, unit variance)
    vel_indices = [F_U1, F_V1, F_W1, F_U2, F_V2, F_W2, F_U3, F_V3, F_W3]
    vel_data = all_fluid[:, vel_indices]
    vel_std = vel_data.std(axis=0) + 1e-12
    vel_global_std = float(np.abs(vel_data).max())  # single scale for all velocities
    for i, idx in enumerate(vel_indices):
        scales[idx] = {"type": "standardize", "mean": 0.0, "std": vel_global_std}

    # Pressure
    p_data = all_fluid[:, F_P]
    scales[F_P] = {"type": "standardize", "mean": 0.0, "std": float(np.abs(p_data).max() + 1e-12)}

    # Temperature: min-max
    T_data = all_fluid[:, F_T]
    scales[F_T] = {"type": "minmax", "min": float(T_data.min()), "max": float(T_data.max())}

    # Turbulence: log-minmax  (k and ω span many orders of magnitude)
    for idx in [F_K, F_OMEGA]:
        data = np.log1p(np.abs(all_fluid[:, idx]) + 1e-20)
        scales[idx] = {"type": "log_minmax", "min": float(data.min()), "max": float(data.max())}

    # VOF: [0,1] range, light scaling
    for idx in [F_VOF2, F_VOF3]:
        scales[idx] = {"type": "minmax", "min": 0.0, "max": 1.0}

    # Species: [0,1] range
    species_indices = [F_Y_H2O2_V, F_Y_H2O_V, F_Y_AIR_V, F_Y_H2O2_L, F_Y_H2O_L]
    for idx in species_indices:
        scales[idx] = {"type": "minmax", "min": 0.0, "max": 1.0}

    return scales


def _normalize_fluid(fd: np.ndarray, scales: dict) -> np.ndarray:
    """Normalize a (n_fluid, N_OUTPUT_FIELDS) array in-place."""
    out = fd.copy()
    for idx, s in scales.items():
        if s["type"] == "standardize":
            out[:, idx] = (fd[:, idx] - s["mean"]) / s["std"]
        elif s["type"] == "minmax":
            rng = s["max"] - s["min"] + 1e-12
            out[:, idx] = (fd[:, idx] - s["min"]) / rng
        elif s["type"] == "log_minmax":
            rng = s["max"] - s["min"] + 1e-12
            out[:, idx] = (np.log1p(np.abs(fd[:, idx]) + 1e-20) - s["min"]) / rng
    return out


# ── Training data builders ────────────────────────────────────

def build_training_data(mesh: dict, dat_dir: Path, dat_glob: str,
                        use_discrete_time_input: bool = True) -> dict:
    """Build single-case training data (20-field fluid + solid T)."""
    cell_centers = compute_cell_centers(mesh)
    times, fluid_data, solid_temp, ps = load_multi_fluid_fields(dat_dir, dat_glob)
    n_fluid, n_solid = ps["n_fluid"], ps["n_solid"]
    n_total = cell_centers.shape[0]
    n_time = len(times)

    print(f"  n_fluid={n_fluid}, n_solid={n_solid}, n_total={n_total}, n_time={n_time}")

    fluid_coords = cell_centers[:n_fluid]
    solid_coords = cell_centers[n_fluid:n_fluid + n_solid] if n_solid > 0 else np.zeros((0, 3))

    all_coords = np.concatenate([fluid_coords, solid_coords], axis=0) if n_solid > 0 else fluid_coords
    coord_min = all_coords.min(axis=0)
    coord_max = all_coords.max(axis=0)
    coord_range = coord_max - coord_min

    chain_scales = coord_range.copy()
    time_scale = float(times.max() - times.min())

    # Normalize coordinates
    norm_fluid_coords = (fluid_coords - coord_min) / (coord_range + 1e-12)
    norm_solid_coords = (solid_coords - coord_min) / (coord_range + 1e-12) if n_solid > 0 else np.zeros((0, 3))

    if use_discrete_time_input:
        norm_times = np.linspace(0.0, 1.0, n_time, dtype=np.float64)
    else:
        norm_times = (times - times.min()) / (time_scale + 1e-12)

    # Compute scales from all fluid data
    all_fluid = np.concatenate(fluid_data, axis=0)
    scales = _compute_scales(all_fluid)

    # Normalize
    norm_fluid_data = [_normalize_fluid(fd, scales) for fd in fluid_data]
    T_min = scales[F_T]["min"]
    T_range = scales[F_T]["max"] - T_min + 1e-12

    # Solid temperature normalization
    if n_solid > 0 and solid_temp[0].size > 0:
        norm_solid_T = [(st - T_min) / T_range for st in solid_temp]
    else:
        norm_solid_T = []

    T_preheat_K, T_h2o2_K = parse_temperatures_from_path(dat_dir)
    T_min = min(T_min, T_preheat_K, T_h2o2_K, 300.0)
    T_max_val = max(scales[F_T]["max"], T_preheat_K, T_h2o2_K, 500.0)
    T_range = T_max_val - T_min + 1e-12
    scales[F_T]["min"] = T_min
    scales[F_T]["max"] = T_max_val

    all_norm_coords = np.concatenate([norm_fluid_coords, norm_solid_coords], axis=0) if n_solid > 0 else norm_fluid_coords
    solid_mask_all = np.concatenate([
        np.zeros(n_fluid, dtype=bool),
        np.ones(n_solid, dtype=bool)
    ]) if n_solid > 0 else np.zeros(n_fluid, dtype=bool)

    return {
        "cell_centers": cell_centers,
        "fluid_coords": fluid_coords,
        "solid_coords": solid_coords,
        "T_preheat_K": T_preheat_K,
        "T_h2o2_K": T_h2o2_K,
        "norm_coords": norm_fluid_coords,
        "norm_coords_all": all_norm_coords,
        "solid_mask_all": solid_mask_all,
        "norm_times": norm_times,
        "times": times,
        "fluid_data": fluid_data,
        "solid_temp": solid_temp,
        "norm_fluid_data": norm_fluid_data,
        "norm_solid_T": norm_solid_T,
        "coord_min": coord_min,
        "coord_max": coord_max,
        "coord_range": coord_range,
        "chain_scales": chain_scales,
        "time_scale": time_scale,
        "scales": scales,
        "T_min": T_min,
        "T_range": T_range,
        "n_cells": n_total,
        "n_fluid": n_fluid,
        "n_solid": n_solid,
        "n_time": n_time,
        "field_names": FIELD_NAMES,
    }


def build_multi_case_training_data(mesh, data_root: Path, dat_glob: str,
                                   case_glob: str = "T_*_*",
                                   use_discrete_time_input: bool = True) -> dict:
    """Build multi-case training data with 20-field multi-fluid layout."""
    cell_centers = compute_cell_centers(mesh)
    n_total_cells = cell_centers.shape[0]
    case_dirs = discover_case_dirs(data_root, dat_glob, case_glob)
    raw_cases = []

    print(f"  Found {len(case_dirs)} case folders")
    for case_dir in case_dirs:
        T_preheat_K, T_h2o2_K = parse_temperatures_from_path(case_dir)
        times, fluid_data, solid_temp, ps = load_multi_fluid_fields(case_dir, dat_glob)
        raw_cases.append({
            "case_dir": case_dir,
            "case_name": case_dir.name,
            "T_preheat_K": T_preheat_K,
            "T_h2o2_K": T_h2o2_K,
            "times": times,
            "fluid_data": fluid_data,
            "solid_temp": solid_temp,
            "n_fluid": ps["n_fluid"],
            "n_solid": ps["n_solid"],
            "n_time": len(times),
        })

    n_fluid = raw_cases[0]["n_fluid"]
    n_solid = raw_cases[0]["n_solid"]
    for case in raw_cases:
        if case["n_fluid"] != n_fluid or case["n_solid"] != n_solid:
            raise ValueError(
                f"Case {case['case_name']} mesh field size differs: "
                f"fluid={case['n_fluid']} solid={case['n_solid']}"
            )

    fluid_coords = cell_centers[:n_fluid]
    solid_coords = cell_centers[n_fluid:n_fluid + n_solid] if n_solid > 0 else np.zeros((0, 3))
    all_coords = np.concatenate([fluid_coords, solid_coords], axis=0) if n_solid > 0 else fluid_coords
    coord_min = all_coords.min(axis=0)
    coord_max = all_coords.max(axis=0)
    coord_range = coord_max - coord_min
    chain_scales = coord_range.copy()

    all_times = np.concatenate([case["times"] for case in raw_cases])
    time_min = float(all_times.min())
    time_max = float(all_times.max())
    time_scale = time_max - time_min

    norm_fluid_coords = (fluid_coords - coord_min) / (coord_range + 1e-12)
    norm_solid_coords = (solid_coords - coord_min) / (coord_range + 1e-12) if n_solid > 0 else np.zeros((0, 3))
    all_norm_coords = np.concatenate([norm_fluid_coords, norm_solid_coords], axis=0) if n_solid > 0 else norm_fluid_coords
    solid_mask_all = np.concatenate([
        np.zeros(n_fluid, dtype=bool),
        np.ones(n_solid, dtype=bool),
    ]) if n_solid > 0 else np.zeros(n_fluid, dtype=bool)

    # Global scales across all cases
    all_fluid = np.concatenate([
        np.concatenate(case["fluid_data"], axis=0)
        for case in raw_cases
    ], axis=0)
    scales = _compute_scales(all_fluid)
    T_min = scales[F_T]["min"]
    T_max_val = scales[F_T]["max"]

    # Expand T range with inlet temperatures
    inlet_temps = np.array([
        v for case in raw_cases
        for v in (case["T_preheat_K"], case["T_h2o2_K"])
    ])
    T_min = min(T_min, inlet_temps.min(), 300.0)
    T_max_val = max(T_max_val, inlet_temps.max(), 500.0)
    T_range = T_max_val - T_min + 1e-12
    scales[F_T]["min"] = T_min
    scales[F_T]["max"] = T_max_val

    def normalize(fd):
        return _normalize_fluid(fd, scales)

    cases = []
    for case in raw_cases:
        if use_discrete_time_input:
            norm_times = np.linspace(0.0, 1.0, case["n_time"], dtype=np.float64)
        else:
            norm_times = (case["times"] - time_min) / (time_scale + 1e-12)
        norm_fluid_data = [normalize(fd) for fd in case["fluid_data"]]
        if case["n_solid"] > 0 and case["solid_temp"][0].size > 0:
            norm_solid_T = [(st - T_min) / T_range for st in case["solid_temp"]]
        else:
            norm_solid_T = []

        bc_params_norm = np.array([
            (case["T_preheat_K"] - T_min) / T_range,
            (case["T_h2o2_K"] - T_min) / T_range,
        ], dtype=np.float32)

        inlet_a_T, inlet_b_T = make_inlet_temperature_tables(
            case["times"], case["T_preheat_K"], case["T_h2o2_K"])
        cases.append({
            **case,
            "norm_times": norm_times,
            "norm_fluid_data": norm_fluid_data,
            "norm_solid_T": norm_solid_T,
            "bc_params_norm": bc_params_norm,
            "inlet_a_T_norm": ((inlet_a_T - T_min) / T_range).astype(np.float32),
            "inlet_b_T_norm": ((inlet_b_T - T_min) / T_range).astype(np.float32),
        })

    first = cases[0]
    # Classify solid cells for targeted surface supervision
    solid_classification = classify_solid_cells(mesh, n_solid)
    print(f"  Solid classification: fluid_i_soild={len(solid_classification['fluid_i_soild']):,}, "
          f"fluid_o_soild={len(solid_classification['fluid_o_soild']):,}, "
          f"soild_boundary={len(solid_classification['soild_boundary']):,}, "
          f"interior={len(solid_classification['interior']):,}")

    # Pre-compute fluid_i-soild face centers (on the actual interface surface)
    fluid_i_soild_face_data = compute_fluid_i_soild_face_data(
        mesh, n_solid, solid_classification, coord_min, coord_range)
    print(f"  fluid-solid interface faces: {len(fluid_i_soild_face_data['face_to_cell']):,}")

    # Classify fluid cells: near-wall vs core (for balanced sampling)
    fluid_classification = classify_fluid_cells(mesh, n_fluid)
    print(f"  Fluid classification: fluid_i={len(fluid_classification['fluid_i']):,} "
          f"(core={len(fluid_classification['fluid_i_core']):,}, wall={len(fluid_classification['fluid_i_wall']):,}), "
          f"fluid_o={len(fluid_classification['fluid_o']):,} "
          f"(core={len(fluid_classification['fluid_o_core']):,}, wall={len(fluid_classification['fluid_o_wall']):,})")

    return {
        "cases": cases,
        "n_cases": len(cases),
        "cell_centers": cell_centers,
        "fluid_coords": fluid_coords,
        "solid_coords": solid_coords,
        "solid_classification": solid_classification,
        "fluid_classification": fluid_classification,  # near-wall vs core
        "fluid_i_soild_face_data": fluid_i_soild_face_data,  # face centers on interface
        "T_preheat_K": first["T_preheat_K"],
        "T_h2o2_K": first["T_h2o2_K"],
        "norm_coords": norm_fluid_coords,
        "norm_coords_all": all_norm_coords,
        "solid_mask_all": solid_mask_all,
        "norm_times": first["norm_times"],
        "times": all_times,
        "fluid_data": first["fluid_data"],
        "solid_temp": first["solid_temp"],
        "norm_fluid_data": first["norm_fluid_data"],
        "norm_solid_T": first["norm_solid_T"],
        "coord_min": coord_min,
        "coord_max": coord_max,
        "coord_range": coord_range,
        "chain_scales": chain_scales,
        "time_min": time_min,
        "time_max": time_max,
        "time_scale": time_scale,
        "scales": scales,
        "T_min": T_min,
        "T_range": T_range,
        "n_cells": n_total_cells,
        "n_fluid": n_fluid,
        "n_solid": n_solid,
        "n_time": max(case["n_time"] for case in cases),
        "field_names": FIELD_NAMES,
    }


# ── Point samplers (20-field aware) ──────────────────────────

import torch  # noqa: E402


def get_fluid_i_soild_cells(mesh: dict, n_solid: int) -> np.ndarray:
    """Identify solid cells adjacent to the inner fluid chamber (fluid_i).

    Uses the fluid_i-soild-shadow face zone (zone id=3) to find solid cells
    whose faces contact the inner fluid zone. These are the critical cells
    for sterilization temperature prediction.

    Returns:
        fluid_i_cells: int32 array of 0-based solid-local cell indices
    """
    face_c0 = mesh["face_c0"]
    zones = mesh.get("zones", {})

    target_zone = None
    for name, info in zones.items():
        if "fluid_i-soild-shadow" in name:
            target_zone = info
            break

    if target_zone is None:
        face_min, face_max = 88120, 118294
    else:
        face_min, face_max = target_zone["face_min"], target_zone["face_max"]

    fluid_i_cells = set()
    for fid in range(face_min - 1, face_max):
        c0 = int(face_c0[fid])
        if 0 <= c0 < n_solid:
            fluid_i_cells.add(c0)

    return np.array(sorted(fluid_i_cells), dtype=np.int32)


def classify_solid_cells(mesh: dict, n_solid: int) -> dict[str, np.ndarray]:
    """Classify all solid cells by adjacent Fluent zone.

    Iterates shadow zone faces directly — shadow face_c0 is the solid-local cell index.
    No offset math needed.

    Returns dict with keys: fluid_i_soild, fluid_o_soild, soild_boundary, interior
    """
    face_c0 = mesh["face_c0"]
    zones = mesh.get("zones", {})
    n_solid_int = int(n_solid)

    fluid_i_soild = set()
    fluid_o_soild = set()
    soild_boundary = set()

    # Shadow zones: face_c0 → solid cell directly
    shadow_zones = [
        ("fluid_i-soild-shadow", fluid_i_soild),
        ("fluid_o-soild-shadow", fluid_o_soild),
    ]

    for shadow_name, target_set in shadow_zones:
        info = zones.get(shadow_name)
        if info is None:
            continue
        for fid in range(info["face_min"] - 1, info["face_max"]):
            c0 = int(face_c0[fid])
            if 0 <= c0 < n_solid_int:
                target_set.add(c0)

    # soild:1 zone — boundary faces, c0 = solid cell directly
    for name, info in zones.items():
        if name == "soild:1":
            for fid in range(info["face_min"] - 1, info["face_max"]):
                c0 = int(face_c0[fid])
                if 0 <= c0 < n_solid_int:
                    soild_boundary.add(c0)
            break

    all_solid = set(range(n_solid_int))
    all_surface = fluid_i_soild | fluid_o_soild | soild_boundary
    interior = all_solid - all_surface

    return {
        "fluid_i_soild": np.array(sorted(fluid_i_soild), dtype=np.int32),
        "fluid_o_soild": np.array(sorted(fluid_o_soild), dtype=np.int32),
        "soild_boundary": np.array(sorted(soild_boundary), dtype=np.int32),
        "interior": np.array(sorted(interior), dtype=np.int32),
    }


def classify_fluid_cells(mesh: dict, n_fluid: int) -> dict[str, np.ndarray]:
    """Classify fluid cells by Fluent cell zone: fluid_i, fluid_o.

    Uses interior face c0 to determine which cells belong to which zone.
    interior--fluid_i faces [2834861, 3732152] → c0 = fluid_i cells
    interior--fluid_o faces [517851, 2834860] → c0 = fluid_o cells

    Additionally tags near-wall cells via solid interface face zones.

    Returns dict: fluid_i, fluid_o, fluid_i_wall, fluid_o_wall (0-based fluid-local).
    """
    face_c0 = mesh["face_c0"]
    zones = mesh.get("zones", {})
    n_fluid_int = int(n_fluid)

    fi_all = set()
    fo_all = set()

    # interior fluid zone faces: c0 identifies owning cell's zone
    for name in zones:
        if name == "interior--fluid_i":
            info = zones[name]
            for fid in range(info["face_min"] - 1, info["face_max"]):
                c0 = int(face_c0[fid])
                if 0 <= c0 < n_fluid_int:
                    fi_all.add(c0)
        elif name == "interior--fluid_o":
            info = zones[name]
            for fid in range(info["face_min"] - 1, info["face_max"]):
                c0 = int(face_c0[fid])
                if 0 <= c0 < n_fluid_int:
                    fo_all.add(c0)
        elif name == "fluid_o:1":
            info = zones[name]
            for fid in range(info["face_min"] - 1, info["face_max"]):
                c0 = int(face_c0[fid])
                if 0 <= c0 < n_fluid_int:
                    fo_all.add(c0)

    # Solid interface faces: tag wall-adjacent cells per zone
    fi_wall = set()
    fo_wall = set()
    for zone_name in ["fluid_i-soild", "fluid_o-soild"]:
        info = zones.get(zone_name)
        if info is None:
            continue
        for fid in range(info["face_min"] - 1, info["face_max"]):
            c0 = int(face_c0[fid])
            if 0 <= c0 < n_fluid_int:
                if zone_name == "fluid_i-soild":
                    fi_wall.add(c0)
                    fi_all.add(c0)
                else:
                    fo_wall.add(c0)
                    fo_all.add(c0)

    # Core = zone cells NOT adjacent to any wall
    fi_core = fi_all - fi_wall
    fo_core = fo_all - fo_wall

    return {
        "fluid_i": np.array(sorted(fi_all), dtype=np.int32),
        "fluid_o": np.array(sorted(fo_all), dtype=np.int32),
        "fluid_i_wall": np.array(sorted(fi_wall), dtype=np.int32),
        "fluid_o_wall": np.array(sorted(fo_wall), dtype=np.int32),
        "fluid_i_core": np.array(sorted(fi_core), dtype=np.int32),
        "fluid_o_core": np.array(sorted(fo_core), dtype=np.int32),
    }


def compute_fluid_i_soild_face_data(mesh: dict, n_solid: int,
                                    classification: dict,
                                    coord_min: np.ndarray,
                                    coord_range: np.ndarray) -> dict:
    """Compute face centers for the complete fluid-solid interface.

    Combines BOTH fluid_i-soild-shadow AND fluid_o-soild-shadow zones.
    They are the two sides of the thin solid wall (~0.2 mm thick) and together
    form the complete fluid-solid interface.

    Returns dict with:
        norm_face_coords: (n_faces, 3) normalized face centers
        face_to_cell:     (n_faces,) solid-local cell index per face
        face_centers_m:   (n_faces, 3) raw face centers in meters
    """
    face_c0 = mesh["face_c0"]
    zones = mesh["zones"]
    node_coords = mesh["node_coords"]
    face_nodes = mesh["face_nodes"]
    offsets = mesh["offsets"]
    n_solid_int = int(n_solid)

    all_face_centers = []
    all_face_to_cell = []

    # Both shadow zones contribute to the fluid-solid interface
    for shadow_name in ["fluid_i-soild-shadow", "fluid_o-soild-shadow"]:
        shadow_info = zones.get(shadow_name)
        if shadow_info is None:
            continue

        fm_shadow = shadow_info["face_min"]
        n_faces = shadow_info["n_faces"]

        for i in range(n_faces):
            fid_shadow = (fm_shadow + i) - 1
            nids = face_nodes[offsets[fid_shadow]:offsets[fid_shadow + 1]]
            c = node_coords[nids - 1].mean(axis=0)
            c0 = int(face_c0[fid_shadow])
            if 0 <= c0 < n_solid_int:
                all_face_centers.append(c)
                all_face_to_cell.append(c0)

    if not all_face_centers:
        return {"norm_face_coords": np.zeros((0, 3), dtype=np.float32),
                "face_to_cell": np.zeros(0, dtype=np.int32),
                "face_centers_m": np.zeros((0, 3), dtype=np.float64)}

    face_centers = np.array(all_face_centers, dtype=np.float64)
    face_to_cell = np.array(all_face_to_cell, dtype=np.int32)
    norm_face_coords = ((face_centers - coord_min) / coord_range).astype(np.float32)

    return {
        "norm_face_coords": norm_face_coords,
        "face_to_cell": face_to_cell,
        "face_centers_m": face_centers,
    }


def _sample_cases(data: dict, n_points: int, rng) -> np.ndarray:
    n_cases = data.get("n_cases", 1)
    return rng.integers(0, n_cases, size=n_points)


def _case_params(case: dict, n_points: int) -> np.ndarray:
    return np.repeat(case["bc_params_norm"][np.newaxis, :], n_points, axis=0)


def make_collocation_points(data: dict, n_points: int, rng,
                            perturb_sigma: float = 0.0) -> tuple:
    coords_all = data["norm_coords_all"]
    solid_mask_all = data["solid_mask_all"]
    n_total = len(coords_all)
    n_fluid = data["n_fluid"]
    fluid_cls = data.get("fluid_classification", {})

    fi_core = fluid_cls.get("fluid_i_core", None)

    if fi_core is not None and len(fi_core) > 0:
        # 30% fluid_i_core (center), 20% all fluid, 50% all cells
        n_fi_core = int(n_points * 0.30)
        n_fl = int(n_points * 0.20)
        n_all = n_points - n_fi_core - n_fl

        cell_idx = np.concatenate([
            rng.choice(fi_core, size=n_fi_core, replace=True),
            rng.integers(0, n_fluid, size=n_fl) if n_fl > 0 else np.array([], dtype=np.int64),
            rng.integers(0, n_total, size=n_all),
        ])
        rng.shuffle(cell_idx)
    else:
        cell_idx = rng.integers(0, n_total, size=n_points)

    case_idx = _sample_cases(data, n_points, rng)
    x = coords_all[cell_idx].copy()
    solid_mask = solid_mask_all[cell_idx].copy()
    t = np.zeros((n_points, 1), dtype=np.float32)
    bc_params = np.zeros((n_points, 2), dtype=np.float32)

    if "cases" in data:
        for case_id, case in enumerate(data["cases"]):
            idx = case_idx == case_id
            if idx.sum() == 0:
                continue
            time_idx = rng.integers(0, case["n_time"], size=idx.sum())
            t[idx, 0] = case["norm_times"][time_idx]
            bc_params[idx] = _case_params(case, idx.sum())
    else:
        time_idx = rng.integers(0, data["n_time"], size=n_points)
        t[:, 0] = data["norm_times"][time_idx]
        bc_params[:] = np.array([
            (data["T_preheat_K"] - data["T_min"]) / data["T_range"],
            (data["T_h2o2_K"] - data["T_min"]) / data["T_range"],
        ], dtype=np.float32)

    if perturb_sigma > 0:
        pert = rng.normal(0, perturb_sigma, size=x.shape)
        x = np.clip(x + pert, 0.0, 1.0)

    return (torch.from_numpy(x).float(),
            torch.from_numpy(t).float(),
            torch.from_numpy(solid_mask).bool(),
            torch.from_numpy(bc_params).float())


def make_data_points(data: dict, n_points: int, rng) -> tuple:
    """Sample fluid data points — weighted toward fluid_i core (center region)."""
    n_fluid = data["n_fluid"]
    fluid_cls = data.get("fluid_classification", {})

    fi_core = fluid_cls.get("fluid_i_core", None)
    fi_wall = fluid_cls.get("fluid_i_wall", None)
    fo_cells = fluid_cls.get("fluid_o", None)

    if fi_core is not None and len(fi_core) > 0:
        n_fi_core = int(n_points * 0.40)
        n_fi_wall = int(n_points * 0.20)
        n_fo = int(n_points * 0.25)
        n_random = n_points - n_fi_core - n_fi_wall - n_fo

        parts = [rng.choice(fi_core, size=n_fi_core, replace=True),
                 rng.choice(fi_wall, size=n_fi_wall, replace=True)]
        if len(fo_cells) > 0:
            parts.append(rng.choice(fo_cells, size=n_fo, replace=True))
        else:
            parts.append(rng.integers(0, n_fluid, size=n_fo))
        parts.append(rng.integers(0, n_fluid, size=n_random))
        cell_idx = np.concatenate(parts)
        rng.shuffle(cell_idx)
    else:
        cell_idx = rng.integers(0, n_fluid, size=n_points)

    case_idx = _sample_cases(data, n_points, rng)
    x = data["norm_coords"][cell_idx]
    t = np.zeros((n_points, 1), dtype=np.float32)
    v = np.zeros((n_points, N_OUTPUT_FIELDS), dtype=np.float32)
    bc_params = np.zeros((n_points, 2), dtype=np.float32)

    if "cases" in data:
        for case_id, case in enumerate(data["cases"]):
            idx = case_idx == case_id
            if idx.sum() == 0:
                continue
            time_idx = rng.integers(0, case["n_time"], size=idx.sum())
            fluid_array = np.array(case["norm_fluid_data"])
            t[idx, 0] = case["norm_times"][time_idx]
            v[idx] = fluid_array[time_idx, cell_idx[idx]]
            bc_params[idx] = _case_params(case, idx.sum())
    else:
        time_idx = rng.integers(0, data["n_time"], size=n_points)
        fluid_array = np.array(data["norm_fluid_data"])
        t[:, 0] = data["norm_times"][time_idx]
        v = fluid_array[time_idx, cell_idx]
        bc_params[:] = np.array([
            (data["T_preheat_K"] - data["T_min"]) / data["T_range"],
            (data["T_h2o2_K"] - data["T_min"]) / data["T_range"],
        ], dtype=np.float32)

    return (torch.from_numpy(x).float(),
            torch.from_numpy(t).float(),
            torch.from_numpy(v).float(),
            torch.from_numpy(bc_params).float())


def make_initial_points(data: dict, n_points: int, rng) -> tuple:
    coords_all = data["norm_coords_all"]
    solid_mask_all = data["solid_mask_all"]
    n_total = len(coords_all)
    cell_idx = rng.integers(0, n_total, size=n_points)
    case_idx = _sample_cases(data, n_points, rng)
    x = coords_all[cell_idx].copy()
    solid_mask = solid_mask_all[cell_idx]
    v = np.zeros((n_points, N_OUTPUT_FIELDS), dtype=np.float32)
    bc_params = np.zeros((n_points, 2), dtype=np.float32)

    cases = data["cases"] if "cases" in data else [data]
    for case_id, case in enumerate(cases):
        idx = case_idx == case_id if "cases" in data else np.ones(n_points, dtype=bool)
        if idx.sum() == 0:
            continue
        fluid_idx = idx & (~solid_mask)
        if fluid_idx.sum() > 0:
            v[fluid_idx] = case["norm_fluid_data"][0][cell_idx[fluid_idx]].copy()
        solid_idx = idx & solid_mask
        if solid_idx.sum() > 0 and data["n_solid"] > 0 and len(case["norm_solid_T"]) > 0:
            solid_local_idx = cell_idx[solid_idx] - data["n_fluid"]
            v[solid_idx, F_T] = case["norm_solid_T"][0][solid_local_idx]
        if "bc_params_norm" in case:
            bc_params[idx] = _case_params(case, idx.sum())
        else:
            bc_params[idx] = np.array([
                (data["T_preheat_K"] - data["T_min"]) / data["T_range"],
                (data["T_h2o2_K"] - data["T_min"]) / data["T_range"],
            ], dtype=np.float32)

    return (torch.from_numpy(x).float(),
            torch.from_numpy(v).float(),
            torch.from_numpy(bc_params).float())


def make_solid_temp_points(data: dict, n_points: int, rng) -> tuple:
    n_solid = data["n_solid"]
    if n_solid == 0 or ("cases" not in data and len(data["norm_solid_T"]) == 0):
        return (torch.zeros(0, 3), torch.zeros(0, 1), torch.zeros(0, 1), torch.zeros(0, 2))

    cell_idx = rng.integers(0, n_solid, size=n_points)
    case_idx = _sample_cases(data, n_points, rng)
    coord_idx = data["n_fluid"] + cell_idx
    x = data["norm_coords_all"][coord_idx]
    t = np.zeros((n_points, 1), dtype=np.float32)
    v_T = np.zeros((n_points, 1), dtype=np.float32)
    bc_params = np.zeros((n_points, 2), dtype=np.float32)

    if "cases" in data:
        for case_id, case in enumerate(data["cases"]):
            idx = case_idx == case_id
            if idx.sum() == 0 or len(case["norm_solid_T"]) == 0:
                continue
            time_idx = rng.integers(0, case["n_time"], size=idx.sum())
            solid_array = np.array(case["norm_solid_T"])
            t[idx, 0] = case["norm_times"][time_idx]
            v_T[idx, 0] = solid_array[time_idx, cell_idx[idx]]
            bc_params[idx] = _case_params(case, idx.sum())
    else:
        time_idx = rng.integers(0, data["n_time"], size=n_points)
        solid_array = np.array(data["norm_solid_T"])
        t[:, 0] = data["norm_times"][time_idx]
        v_T[:, 0] = solid_array[time_idx, cell_idx]
        bc_params[:] = np.array([
            (data["T_preheat_K"] - data["T_min"]) / data["T_range"],
            (data["T_h2o2_K"] - data["T_min"]) / data["T_range"],
        ], dtype=np.float32)

    return (torch.from_numpy(x).float(),
            torch.from_numpy(t).float(),
            torch.from_numpy(v_T).float(),
            torch.from_numpy(bc_params).float())


def make_solid_temp_snapshot(data: dict, case_idx: int = 0, time_idx: int = -1) -> tuple:
    n_solid = data["n_solid"]
    if n_solid == 0:
        return (torch.zeros(0, 3), torch.zeros(0, 1), torch.zeros(0, 1), torch.zeros(0, 2))

    coord_idx = np.arange(data["n_fluid"], data["n_fluid"] + n_solid)
    x = data["norm_coords_all"][coord_idx]
    if "cases" in data:
        case_idx = int(np.clip(case_idx, 0, len(data["cases"]) - 1))
        case = data["cases"][case_idx]
        if len(case["norm_solid_T"]) == 0:
            return (torch.zeros(0, 3), torch.zeros(0, 1), torch.zeros(0, 1), torch.zeros(0, 2))
        time_idx = time_idx % case["n_time"]
        t_value = case["norm_times"][time_idx]
        v_T = np.array(case["norm_solid_T"])[time_idx].reshape(-1, 1)
        bc_params = _case_params(case, n_solid)
    else:
        if len(data["norm_solid_T"]) == 0:
            return (torch.zeros(0, 3), torch.zeros(0, 1), torch.zeros(0, 1), torch.zeros(0, 2))
        time_idx = time_idx % data["n_time"]
        t_value = data["norm_times"][time_idx]
        v_T = np.array(data["norm_solid_T"])[time_idx].reshape(-1, 1)
        bc_params = np.repeat(np.array([[
            (data["T_preheat_K"] - data["T_min"]) / data["T_range"],
            (data["T_h2o2_K"] - data["T_min"]) / data["T_range"],
        ]], dtype=np.float32), n_solid, axis=0)
    t = np.full((n_solid, 1), t_value, dtype=np.float32)
    return (torch.from_numpy(x).float(),
            torch.from_numpy(t).float(),
            torch.from_numpy(v_T.astype(np.float32)).float(),
            torch.from_numpy(bc_params).float())


def make_fluid_i_soild_temp_snapshot(data: dict, case_idx: int = 0, time_idx: int = -1) -> tuple:
    """Full-resolution snapshot of fluid_i-soild interface — face centers (ON the surface)."""
    n_solid = data["n_solid"]
    if n_solid == 0:
        return (torch.zeros(0, 3), torch.zeros(0, 1), torch.zeros(0, 1), torch.zeros(0, 2))

    face_data = data.get("fluid_i_soild_face_data", {})
    face_coords = face_data.get("norm_face_coords", None)
    face_to_cell = face_data.get("face_to_cell", None)

    if face_coords is not None and len(face_coords) > 0:
        x = face_coords
        cell_idx = face_to_cell
    else:
        # Fallback to cell centers
        classification = data.get("solid_classification", {})
        fluid_i_cells = classification.get("fluid_i_soild",
                        np.arange(n_solid, dtype=np.int32))
        x = data["norm_coords_all"][data["n_fluid"] + fluid_i_cells]
        cell_idx = fluid_i_cells

    n_pts = len(x)

    if "cases" in data:
        case_idx = int(np.clip(case_idx, 0, len(data["cases"]) - 1))
        case = data["cases"][case_idx]
        if len(case["norm_solid_T"]) == 0:
            return (torch.zeros(0, 3), torch.zeros(0, 1), torch.zeros(0, 1), torch.zeros(0, 2))
        time_idx = time_idx % case["n_time"]
        t_value = case["norm_times"][time_idx]
        v_T = np.array(case["norm_solid_T"])[time_idx][cell_idx].reshape(-1, 1)
        bc_params = _case_params(case, n_pts)
    else:
        if len(data["norm_solid_T"]) == 0:
            return (torch.zeros(0, 3), torch.zeros(0, 1), torch.zeros(0, 1), torch.zeros(0, 2))
        time_idx = time_idx % data["n_time"]
        t_value = data["norm_times"][time_idx]
        v_T = np.array(data["norm_solid_T"])[time_idx][cell_idx].reshape(-1, 1)
        bc_params = np.repeat(np.array([[
            (data["T_preheat_K"] - data["T_min"]) / data["T_range"],
            (data["T_h2o2_K"] - data["T_min"]) / data["T_range"],
        ]], dtype=np.float32), n_pts, axis=0)
    t = np.full((n_pts, 1), t_value, dtype=np.float32)
    return (torch.from_numpy(x).float(),
            torch.from_numpy(t).float(),
            torch.from_numpy(v_T.astype(np.float32)).float(),
            torch.from_numpy(bc_params).float())


def make_fluid_i_soild_temp_points(data: dict, n_points: int, rng) -> tuple:
    """Generate solid temperature supervision points on the fluid-solid interface.

    Samples 80% from combined face centers (fluid_i-soild + fluid_o-soild shadow zones
    = both sides of the thin solid wall, together forming the complete interface).
    20% from random solid cells for interior coverage.

    Face→cell mapping provides the owner cell's temperature as ground truth.
    """
    n_solid = data["n_solid"]
    if n_solid == 0:
        return (torch.zeros(0, 3), torch.zeros(0, 1), torch.zeros(0, 1), torch.zeros(0, 2))

    face_data = data.get("fluid_i_soild_face_data", {})
    face_coords = face_data.get("norm_face_coords", None)
    face_to_cell = face_data.get("face_to_cell", None)

    if face_coords is not None and len(face_coords) > 0:
        n_faces = len(face_coords)
        n_face_pts = int(n_points * 0.8)
        n_random = n_points - n_face_pts

        # 80% from interface faces (both fluid_i + fluid_o combined)
        face_idx = rng.choice(n_faces, size=n_face_pts, replace=True)

        # 20% random solid cells for interior coverage
        random_cell_idx = rng.integers(0, n_solid, size=n_random)

        # Coordinates: face centers for interface samples, cell centers for interior
        x = np.zeros((n_points, 3), dtype=np.float32)
        x[:n_face_pts] = face_coords[face_idx]
        for i, cid in enumerate(random_cell_idx):
            x[n_face_pts + i] = data["norm_coords_all"][data["n_fluid"] + cid]

        # Cell index for temperature lookup:
        # face samples → owner cell; interior samples → self
        cell_idx = np.zeros(n_points, dtype=np.int32)
        cell_idx[:n_face_pts] = face_to_cell[face_idx]
        cell_idx[n_face_pts:] = random_cell_idx

        # Shuffle so face/cell samples are interleaved
        perm = rng.permutation(n_points)
        x = x[perm]
        cell_idx = cell_idx[perm]
    else:
        # Fallback: uniform cell sampling
        cell_idx = rng.integers(0, n_solid, size=n_points)
        x = data["norm_coords_all"][data["n_fluid"] + cell_idx]

    case_idx = _sample_cases(data, n_points, rng)
    t = np.zeros((n_points, 1), dtype=np.float32)
    v_T = np.zeros((n_points, 1), dtype=np.float32)
    bc_params = np.zeros((n_points, 2), dtype=np.float32)

    if "cases" in data:
        for case_id, case in enumerate(data["cases"]):
            idx = case_idx == case_id
            if idx.sum() == 0 or len(case["norm_solid_T"]) == 0:
                continue
            time_idx = rng.integers(0, case["n_time"], size=idx.sum())
            solid_array = np.array(case["norm_solid_T"])
            t[idx, 0] = case["norm_times"][time_idx]
            v_T[idx, 0] = solid_array[time_idx, cell_idx[idx]]
            bc_params[idx] = _case_params(case, idx.sum())
    else:
        time_idx = rng.integers(0, data["n_time"], size=n_points)
        solid_array = np.array(data["norm_solid_T"])
        t[:, 0] = data["norm_times"][time_idx]
        v_T[:, 0] = solid_array[time_idx, cell_idx]
        bc_params[:] = np.array([
            (data["T_preheat_K"] - data["T_min"]) / data["T_range"],
            (data["T_h2o2_K"] - data["T_min"]) / data["T_range"],
        ], dtype=np.float32)

    return (torch.from_numpy(x).float(),
            torch.from_numpy(t).float(),
            torch.from_numpy(v_T).float(),
            torch.from_numpy(bc_params).float())


# ── BC helpers ────────────────────────────────────────────────

def make_inlet_temperature_tables(times: np.ndarray, T_preheat: float, T_h2o2: float,
                                  T_drying: float = 413.15,
                                  T_default: float = 300.0) -> tuple[np.ndarray, np.ndarray]:
    inlet_a = np.full_like(times, T_default, dtype=np.float64)
    inlet_b = np.full_like(times, T_default, dtype=np.float64)
    inlet_a[(times >= 0.32) & (times < 0.9)] = T_preheat
    inlet_a[(times >= 2.12) & (times < 2.7)] = T_h2o2
    inlet_a[((times >= 4.82) & (times < 5.4)) | ((times >= 6.62) & (times < 7.4))] = T_drying
    inlet_b[(times >= 1.22) & (times < 1.8)] = T_preheat
    inlet_b[(times >= 3.02) & (times < 3.6)] = T_h2o2
    inlet_b[(times >= 5.72) & (times < 6.3)] = T_drying
    return inlet_a, inlet_b


def make_inlet_species_tables(times: np.ndarray, Y_h2o2: float = 0.024989,
                               Y_h2o: float = 0.152936) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build time-dependent species mass fraction profiles for both inlets.

    Species injection only during H2O2 phase (times 2.12-2.7 for inlet_a, 3.02-3.6 for inlet_b).
    """
    n = len(times)
    Y_h2o2_a = np.zeros(n, dtype=np.float32)
    Y_h2o_a  = np.zeros(n, dtype=np.float32)
    Y_h2o2_b = np.zeros(n, dtype=np.float32)
    Y_h2o_b  = np.zeros(n, dtype=np.float32)

    Y_h2o2_a[(times >= 2.12) & (times < 2.7)] = Y_h2o2
    Y_h2o_a[(times >= 2.12) & (times < 2.7)]  = Y_h2o
    Y_h2o2_b[(times >= 3.02) & (times < 3.6)] = Y_h2o2
    Y_h2o_b[(times >= 3.02) & (times < 3.6)]  = Y_h2o

    return Y_h2o2_a, Y_h2o_a, Y_h2o2_b, Y_h2o_b


def make_bc_points(data: dict, n_points: int, rng) -> tuple:
    case_idx = _sample_cases(data, n_points, rng)
    t = np.zeros((n_points, 1), dtype=np.float32)
    bc_params = np.zeros((n_points, 2), dtype=np.float32)
    T_a = np.zeros((n_points, 1), dtype=np.float32)
    T_b = np.zeros((n_points, 1), dtype=np.float32)

    for case_id, case in enumerate(data["cases"]):
        idx = case_idx == case_id
        if idx.sum() == 0:
            continue
        inlet_a = case["inlet_a_T_norm"]
        inlet_b = case["inlet_b_T_norm"]
        default_a = inlet_a[0]
        default_b = inlet_b[0]
        active_idx = np.flatnonzero((np.abs(inlet_a - default_a) > 1e-6) | (np.abs(inlet_b - default_b) > 1e-6))
        change_idx = np.flatnonzero((np.abs(np.diff(inlet_a)) > 1e-6) | (np.abs(np.diff(inlet_b)) > 1e-6))
        if change_idx.size > 0:
            nearby = np.concatenate([change_idx, change_idx + 1, change_idx + 2, np.maximum(change_idx - 1, 0)])
            active_idx = np.unique(np.concatenate([active_idx, nearby[(nearby >= 0) & (nearby < case["n_time"])]]))
        n_active = int(idx.sum() * 0.8) if active_idx.size > 0 else 0
        n_default = idx.sum() - n_active
        time_parts = []
        if n_active > 0:
            time_parts.append(rng.choice(active_idx, size=n_active, replace=True))
        if n_default > 0:
            time_parts.append(rng.integers(0, case["n_time"], size=n_default))
        time_idx = np.concatenate(time_parts) if time_parts else np.zeros(0, dtype=np.int64)
        rng.shuffle(time_idx)
        t[idx, 0] = case["norm_times"][time_idx]
        bc_params[idx] = _case_params(case, idx.sum())
        T_a[idx, 0] = case["inlet_a_T_norm"][time_idx]
        T_b[idx, 0] = case["inlet_b_T_norm"][time_idx]

    return (torch.from_numpy(t).float(),
            torch.from_numpy(bc_params).float(),
            torch.from_numpy(T_a).float(),
            torch.from_numpy(T_b).float())


def split_case_data(data: dict, val_ratio: float, rng) -> tuple[dict, dict | None]:
    if "cases" not in data or data["n_cases"] <= 1 or val_ratio <= 0:
        return data, None
    n_cases = data["n_cases"]
    case_order = rng.permutation(n_cases)
    n_val = int(round(n_cases * val_ratio))
    n_val = min(max(n_val, 1), n_cases - 1)
    val_ids = set(case_order[:n_val].tolist())
    train_cases = [case for idx, case in enumerate(data["cases"]) if idx not in val_ids]
    val_cases = [case for idx, case in enumerate(data["cases"]) if idx in val_ids]

    def make_subset(cases: list[dict]) -> dict:
        subset = dict(data)
        first = cases[0]
        subset["cases"] = cases
        subset["n_cases"] = len(cases)
        subset["T_preheat_K"] = first["T_preheat_K"]
        subset["T_h2o2_K"] = first["T_h2o2_K"]
        subset["norm_times"] = first["norm_times"]
        subset["fluid_data"] = first["fluid_data"]
        subset["solid_temp"] = first["solid_temp"]
        subset["norm_fluid_data"] = first["norm_fluid_data"]
        subset["norm_solid_T"] = first["norm_solid_T"]
        subset["n_time"] = max(case["n_time"] for case in cases)
        subset["case_names"] = [case["case_name"] for case in cases]
        return subset

    return make_subset(train_cases), make_subset(val_cases)


def discover_case_dirs(data_root: Path, dat_glob: str, case_glob: str = "T_*_*") -> list[Path]:
    case_dirs = sorted(p for p in data_root.glob(case_glob) if p.is_dir() and any(p.glob(dat_glob)))
    if case_dirs:
        return case_dirs
    if any(data_root.glob(dat_glob)):
        return [data_root]
    raise FileNotFoundError(f"No case folders under {data_root} matching {case_glob}/{dat_glob}")


def load_inlet_coords(inlet_coords_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    for ext, loader in [(".npy", np.load), (".csv", lambda p: np.loadtxt(p, delimiter=","))]:
        a_path = inlet_coords_dir / f"inlet_a_coords{ext}"
        b_path = inlet_coords_dir / f"inlet_b_coords{ext}"
        if a_path.exists() and b_path.exists():
            return loader(a_path), loader(b_path)
    raise FileNotFoundError(
        f"Inlet coordinate files not found in {inlet_coords_dir}. "
    )


# BC extraction helpers preserved for future use
def extract_face_field(dat_path: Path, field_name: str) -> dict:
    result = {}
    with h5py.File(dat_path, "r") as f:
        base = f"results/1/phase-1/faces/{field_name}"
        if base not in f:
            return {}
        grp = f[base]
        for k in sorted(grp.keys()):
            result[k] = np.asarray(grp[k][()])
    return result
