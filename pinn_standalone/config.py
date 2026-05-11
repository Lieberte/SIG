"""
PINN configuration for H2O2 sterilization — Multi-fluid Eulerian-Eulerian model.

3-phase VOF multiphase:
  Phase 1 — carrier gas (air), with k-ω turbulence
  Phase 2 — vapor (H2O2/H2O/air vapour)
  Phase 3 — liquid (H2O2/H2O droplets)
  Solid zone — aluminium wall (conduction only)

Field layout (19 fields for fluid, 1 for solid):
  [0:3]   u1, v1, w1     Phase-1 velocity (carrier)
  [3:6]   u2, v2, w2     Phase-2 velocity (vapor)
  [6:9]   u3, v3, w3     Phase-3 velocity (liquid)
  [9]     p               Shared pressure
  [10]    T               Shared temperature (thermal equilibrium)
  [11:13] k, omega        Turbulence (phase-1)
  [13:15] vof2, vof3      Volume fractions (vof1 = 1 - vof2 - vof3)
  [15:18] Y_h2o2_v, Y_h2o_v, Y_air_v   Phase-2 species
  [18:20] Y_h2o2_l, Y_h2o_l            Phase-3 species
"""
from dataclasses import dataclass, field
from pathlib import Path


def default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


# ── field index constants ────────────────────────────────────
# Phase 1 (carrier gas)
F_U1, F_V1, F_W1 = 0, 1, 2
# Phase 2 (vapor)
F_U2, F_V2, F_W2 = 3, 4, 5
# Phase 3 (liquid)
F_U3, F_V3, F_W3 = 6, 7, 8
# Shared scalars
F_P = 9
F_T = 10
# Turbulence
F_K, F_OMEGA = 11, 12
# Volume fractions  (vof1 derived = 1 - vof2 - vof3)
F_VOF2, F_VOF3 = 13, 14
# Phase-2 species (vapor)
F_Y_H2O2_V, F_Y_H2O_V, F_Y_AIR_V = 15, 16, 17
# Phase-3 species (liquid)
F_Y_H2O2_L, F_Y_H2O_L = 18, 19

N_OUTPUT_FIELDS = 20

FIELD_NAMES = [
    "u1", "v1", "w1",
    "u2", "v2", "w2",
    "u3", "v3", "w3",
    "p", "T",
    "k", "omega",
    "vof2", "vof3",
    "Y_h2o2_v", "Y_h2o_v", "Y_air_v",
    "Y_h2o2_l", "Y_h2o_l",
]

# Velocity slices per phase
VEL_SLICES = {
    1: (F_U1, F_V1, F_W1),
    2: (F_U2, F_V2, F_W2),
    3: (F_U3, F_V3, F_W3),
}

# Species slices per phase
SPECIES_SLICES = {
    2: [F_Y_H2O2_V, F_Y_H2O_V, F_Y_AIR_V],  # 3 species in vapor
    3: [F_Y_H2O2_L, F_Y_H2O_L],               # 2 species in liquid
}

# Bulk species index per phase (the last one, for closure ΣY=1)
BULK_SPECIES = {2: F_Y_AIR_V, 3: F_Y_H2O_L}

# SV_Y column index in Fluent H5 files → config field index
# Verified against UDF: mass_fraction_h2o2=0.024989, mass_fraction_h2o=0.152936
# Phase 2 (vapor): SV_Y col 0 = h2o2, col 1 = h2o, col 2 = air(bulk)
# Phase 3 (liquid): SV_Y col 0 = h2o2<l>, col 1 = h2o<l>(bulk)
SV_Y_TO_FIELD_P2 = {0: F_Y_H2O2_V, 1: F_Y_H2O_V, 2: F_Y_AIR_V}
SV_Y_TO_FIELD_P3 = {0: F_Y_H2O2_L, 1: F_Y_H2O_L}


@dataclass
class DataConfig:
    """Paths and data extraction settings."""
    _project_root: Path = Path(__file__).resolve().parent.parent
    cas_path: Path = _project_root / "data/meshData/KMB_phase_change_test.cas.h5"
    data_root: Path = _project_root / "data"
    case_glob: str = "T_*_*"
    dat_dir: Path = _project_root / "data/T_160_200"
    dat_glob: str = "*.dat.h5"

    inlet_a_range: tuple[int, int] = (118295, 118467)
    inlet_b_range: tuple[int, int] = (118468, 118638)
    outlet_face_range: tuple[int, int] = (118639, 128110)
    wall_face_ranges: list[tuple[int, int]] = field(default_factory=lambda: [(56737, 57944)])

    solid_zone_name: str = "soild:1"

    inlet_sub_idx: int = 3

    cell_field_names: list[str] = field(default_factory=lambda: [
        "velocityX", "velocityY", "velocityZ", "pressure", "temperature"
    ])

    inlet_coords_path: Path = _project_root / "data/meshData"


@dataclass
class PINNConfig:
    # ── Network ──
    n_spatial_dim: int = 3
    n_output_fields: int = N_OUTPUT_FIELDS
    hidden_size: int = 256          # increased for multi-fluid
    n_hidden_layers: int = 6        # increased for multi-fluid
    activation: str = "tanh"
    parametric_bc: bool = True

    T_preheat_default: float = 433.15   # 160 °C
    T_h2o2_default: float = 473.15      # 200 °C
    T_drying: float = 413.15            # 140 °C

    # ── Inlet species mass fractions (from UDF) ──
    Y_h2o2_inlet: float = 0.024989
    Y_h2o_inlet: float = 0.152936
    Y_air_inlet: float = 1.0 - 0.024989 - 0.152936  # ~0.822

    parametric_training: bool = True
    use_discrete_time_input: bool = True
    hard_inlet_bc: bool = True
    val_case_ratio: float = 0.3
    n_val_data_points: int = 15000
    n_val_solid_temp_points: int = 5000

    # ── Training ──
    device: str = field(default_factory=default_device)
    learning_rate: float = 1e-4
    n_epochs: int = 3000
    batch_size_colloc: int = 512
    batch_size_boundary: int = 512
    batch_size_data: int = 512

    n_collocation: int = 20000
    n_collocation_points: int = 20000
    n_data_points: int = 80000
    n_initial_points: int = 8000
    n_solid_temp_points: int = 8000
    solid_viz_interval: int = 3
    solid_viz_case_idx: int = 0
    solid_viz_time_idx: int = -1
    n_bc_points: int = 4000
    batch_size: int = 512
    collocation_perturb: float = 0.001
    bc_loss_interval: int = 1

    colloc_fluid_ratio: float = 0.7

    # ── Loss weights ──
    lambda_data: float = 5.0
    lambda_physics_fluid: float = 1.0
    lambda_physics_solid: float = 1.0
    lambda_solid_temp: float = 10.0  # inner surface T supervision (sterilization-critical)
    lambda_bc_inlet: float = 50.0
    lambda_bc_wall: float = 5.0
    lambda_initial: float = 5.0
    lambda_species: float = 3.0         # species transport loss weight
    lambda_vof: float = 1.0            # VOF advection loss weight
    lambda_turbulence: float = 0.5     # k-omega turbulence loss weight
    lambda_drag: float = 1.0           # inter-phase drag coupling

    physics_loss_weights: dict[str, float] = field(default_factory=lambda: {
        # Phase-1 PDEs (carrier gas)
        "continuity": 1e-1,
        "momentum_p1_x": 1e-1,
        "momentum_p1_y": 1e-1,
        "momentum_p1_z": 1e-1,
        "k_transport": 5e-2,
        "omega_transport": 5e-2,
        # Phase-2 PDEs (vapor)
        "momentum_p2_x": 1e-1,
        "momentum_p2_y": 1e-1,
        "momentum_p2_z": 1e-1,
        "energy_p2": 1e-1,
        "species_h2o2_v": 1e-1,
        "species_h2o_v": 1e-1,
        "vof_advection": 5e-2,
        # Phase-3 PDEs (liquid)
        "momentum_p3_x": 1e-1,
        "momentum_p3_y": 1e-1,
        "momentum_p3_z": 1e-1,
        "energy_p3": 1e-1,
        "species_h2o2_l": 1e-1,
        # Solid
        "energy_solid": 1e-1,
    })

    # ── Phase-1 (carrier gas — air) ──
    rho_p1: float = 1.1455
    mu_p1: float = 1.879e-5
    cp_p1: float = 1006.0
    k_p1: float = 0.026

    # ── Phase-2 (vapor mixture — H2O2/H2O/air) ──
    rho_p2: float = 1.1455
    mu_p2: float = 1.879e-5
    cp_p2: float = 1500.0
    k_p2: float = 0.02

    # ── Phase-3 (liquid — H2O2/H2O solution) ──
    rho_p3: float = 998.2
    mu_p3: float = 1.003e-3
    cp_p3: float = 4180.0
    k_p3: float = 0.6

    # Species diffusivities (vapor phase)
    D_h2o2_v: float = 2.5e-5
    D_h2o_v: float = 2.5e-5

    # Species diffusivities (liquid phase)
    D_h2o2_l: float = 1.0e-9

    # Inter-phase drag coefficients
    K_drag_12: float = 1.0e4   # phase-1 ↔ phase-2
    K_drag_13: float = 5.0e4   # phase-1 ↔ phase-3

    # ── Solid (polymer / non-metallic wall) ──
    rho_solid: float = 1000.0
    cp_solid: float = 1500.0
    k_solid: float = 0.114

    # ── PDE enabling ──
    enabled_pdes: set[str] = field(default_factory=lambda: {
        "continuity",
        "momentum_p1_x", "momentum_p1_y", "momentum_p1_z",
        "momentum_p2_x", "momentum_p2_y", "momentum_p2_z",
        "momentum_p3_x", "momentum_p3_y", "momentum_p3_z",
        "energy_p2", "energy_p3",
        "energy_solid",
        "k_transport", "omega_transport",
        "species_h2o2_v", "species_h2o_v",
        "species_h2o2_l",
        "vof_advection",
    })

    patience: int = 3000

    output_dir: Path = Path(__file__).resolve().parent / "output"
