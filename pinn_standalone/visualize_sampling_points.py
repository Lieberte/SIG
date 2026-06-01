"""Visualize actual sampling point distribution — Gaussian scatter plots."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DataConfig, PINNConfig
from data_loader import (
    load_cas, compute_cell_centers, classify_solid_cells, classify_fluid_cells,
    load_multi_fluid_fields, compute_fluid_i_soild_face_data,
    make_data_points, make_collocation_points, make_fluid_i_soild_temp_points,
)

dc = DataConfig()
pc = PINNConfig()
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading mesh...")
mesh = load_cas(dc.cas_path)
cell_centers_all = compute_cell_centers(mesh)

dat_files = sorted(dc.dat_dir.glob("*.dat.h5"))
_, _, _, ps = load_multi_fluid_fields(dc.dat_dir, "*.dat.h5")
n_fluid = ps["n_fluid"]
n_solid = ps["n_solid"]
print(f"  fluid={n_fluid:,}  solid={n_solid:,}")

# Classify
solid_cls = classify_solid_cells(mesh, n_solid)
fluid_cls = classify_fluid_cells(mesh, n_fluid, n_solid)

print(f"  fluid_i={len(fluid_cls['fluid_i']):,} (core={len(fluid_cls['fluid_i_core']):,}, wall={len(fluid_cls['fluid_i_wall']):,})")
print(f"  fluid_o={len(fluid_cls['fluid_o']):,} (core={len(fluid_cls['fluid_o_core']):,}, wall={len(fluid_cls['fluid_o_wall']):,})")

# Build minimal data dict for sampling
fluid_coords = cell_centers_all[n_solid + 1 : n_solid + 1 + n_fluid]
coord_min = fluid_coords.min(axis=0)
coord_max = fluid_coords.max(axis=0)
coord_range = coord_max - coord_min + 1e-12

norm_fluid_coords = (fluid_coords - coord_min) / coord_range
solid_coords = cell_centers_all[1 : n_solid + 1]
norm_solid_coords = (solid_coords - coord_min) / coord_range
all_norm_coords = np.vstack([norm_fluid_coords, norm_solid_coords])
solid_mask_all = np.concatenate([np.zeros(n_fluid, dtype=bool), np.ones(n_solid, dtype=bool)])

face_data = compute_fluid_i_soild_face_data(mesh, n_solid, solid_cls, coord_min, coord_range)

rng = np.random.default_rng(42)

n_time = 1
data = {
    "n_fluid": n_fluid,
    "n_solid": n_solid,
    "n_time": n_time,
    "norm_times": np.array([0.5], dtype=np.float32),
    "T_preheat_K": 433.15,
    "T_h2o2_K": 473.15,
    "T_min": 300.0,
    "T_range": 200.0,
    "norm_coords": norm_fluid_coords.astype(np.float32),
    "norm_coords_all": all_norm_coords.astype(np.float32),
    "solid_mask_all": solid_mask_all,
    "fluid_classification": fluid_cls,
    "solid_classification": solid_cls,
    "fluid_i_soild_face_data": face_data,
    "norm_fluid_data": np.zeros((n_time, n_fluid, 20), dtype=np.float32),
    "norm_solid_T": np.zeros((n_time, n_solid), dtype=np.float32),
}

# ── Sample ──
print("\nSampling...")
n_data = 80000
n_colloc = 20000
n_solid_pts = 8000

x_data, _, _, _ = make_data_points(data, n_data, rng)
x_col, _, _, _ = make_collocation_points(data, n_colloc, rng)
x_solid, _, _, _ = make_fluid_i_soild_temp_points(data, n_solid_pts, rng)

# Denormalize for plotting in mm
def denorm(x_norm):
    return x_norm.numpy() * coord_range + coord_min

for name, x, count in [("data", x_data, n_data), ("colloc", x_col, n_colloc), ("solid_temp", x_solid, n_solid_pts)]:
    print(f"  {name}: {x.shape[0]:,} points")

# Reference coords for background
fluid_fi = fluid_coords[fluid_cls["fluid_i"]]  # all fluid_i cells
fluid_fo = fluid_coords[fluid_cls["fluid_o"]]
fi_wall_coords = fluid_coords[fluid_cls["fluid_i_wall"]]
fo_wall_coords = fluid_coords[fluid_cls["fluid_o_wall"]]

# Subsample for background
def sub(arr, n=15000):
    return arr[::max(1, len(arr) // n)]

# ── Plot ──
fig = plt.figure(figsize=(24, 18))

# ── Row 1: Data points ──
x_d = denorm(x_data) * 1000  # mm

# 1a: X-Z projection (top view) — data points
ax1 = fig.add_subplot(3, 4, 1)
ax1.scatter(sub(x_d, 20000)[:, 0], sub(x_d, 20000)[:, 2], c="red", s=0.5, alpha=0.6, rasterized=True)
ax1.set_xlabel("X (mm)"); ax1.set_ylabel("Z (mm)")
ax1.set_title(f"Data points ({n_data:,}) — XZ top view")
ax1.set_aspect("equal")

# 1b: Y-Z projection — data points
ax2 = fig.add_subplot(3, 4, 2)
ax2.scatter(sub(x_d, 20000)[:, 1], sub(x_d, 20000)[:, 2], c="red", s=0.5, alpha=0.6, rasterized=True)
ax2.set_xlabel("Y (mm)"); ax2.set_ylabel("Z (mm)")
ax2.set_title(f"Data points — YZ view")
ax2.set_aspect("equal")

# 1c: X-Y projection — data points
ax3 = fig.add_subplot(3, 4, 3)
ax3.scatter(sub(x_d, 20000)[:, 0], sub(x_d, 20000)[:, 1], c="red", s=0.5, alpha=0.6, rasterized=True)
ax3.set_xlabel("X (mm)"); ax3.set_ylabel("Y (mm)")
ax3.set_title(f"Data points — XY view")
ax3.set_aspect("equal")

# 1d: Density heatmap (Y-Z) for data points
ax4 = fig.add_subplot(3, 4, 4)
h = ax4.hist2d(x_d[:, 1], x_d[:, 2], bins=80, cmap="hot", rasterized=True)
ax4.set_xlabel("Y (mm)"); ax4.set_ylabel("Z (mm)")
ax4.set_title(f"Data density — YZ heatmap")
ax4.set_aspect("equal")
plt.colorbar(h[3], ax=ax4, label="count")

# ── Row 2: Collocation points ──
x_c = denorm(x_col) * 1000

ax5 = fig.add_subplot(3, 4, 5)
ax5.scatter(sub(x_c, 15000)[:, 0], sub(x_c, 15000)[:, 2], c="blue", s=0.5, alpha=0.6, rasterized=True)
ax5.set_xlabel("X (mm)"); ax5.set_ylabel("Z (mm)")
ax5.set_title(f"Colloc points ({n_colloc:,}) — XZ top view")
ax5.set_aspect("equal")

ax6 = fig.add_subplot(3, 4, 6)
ax6.scatter(sub(x_c, 15000)[:, 1], sub(x_c, 15000)[:, 2], c="blue", s=0.5, alpha=0.6, rasterized=True)
ax6.set_xlabel("Y (mm)"); ax6.set_ylabel("Z (mm)")
ax6.set_title(f"Colloc points — YZ view")
ax6.set_aspect("equal")

ax7 = fig.add_subplot(3, 4, 7)
ax7.scatter(sub(x_c, 15000)[:, 0], sub(x_c, 15000)[:, 1], c="blue", s=0.5, alpha=0.6, rasterized=True)
ax7.set_xlabel("X (mm)"); ax7.set_ylabel("Y (mm)")
ax7.set_title(f"Colloc points — XY view")
ax7.set_aspect("equal")

ax8 = fig.add_subplot(3, 4, 8)
h2 = ax8.hist2d(x_c[:, 1], x_c[:, 2], bins=80, cmap="Blues", rasterized=True)
ax8.set_xlabel("Y (mm)"); ax8.set_ylabel("Z (mm)")
ax8.set_title(f"Colloc density — YZ heatmap")
ax8.set_aspect("equal")
plt.colorbar(h2[3], ax=ax8, label="count")

# ── Row 3: Solid temp points + comparison ──
x_s = denorm(x_solid) * 1000

ax9 = fig.add_subplot(3, 4, 9)
ax9.scatter(sub(x_s, 8000)[:, 0], sub(x_s, 8000)[:, 2], c="green", s=0.8, alpha=0.7, rasterized=True)
ax9.set_xlabel("X (mm)"); ax9.set_ylabel("Z (mm)")
ax9.set_title(f"Solid temp points ({n_solid_pts:,}) — XZ")
ax9.set_aspect("equal")

ax10 = fig.add_subplot(3, 4, 10)
ax10.scatter(sub(x_s, 8000)[:, 1], sub(x_s, 8000)[:, 2], c="green", s=0.8, alpha=0.7, rasterized=True)
ax10.set_xlabel("Y (mm)"); ax10.set_ylabel("Z (mm)")
ax10.set_title(f"Solid temp points — YZ")
ax10.set_aspect("equal")

# 11: Overlay: fluid_i outline + data points (YZ) to check wall concentration
ax11 = fig.add_subplot(3, 4, 11)
# Background: fluid_i wall cells (to check if data points cluster there)
fi_w = sub(fi_wall_coords, 5000) * 1000
fo_w = sub(fo_wall_coords, 5000) * 1000
ax11.scatter(fi_w[:, 1], fi_w[:, 2], c="orange", s=2, alpha=0.4, marker="s", label="fi_wall")
ax11.scatter(fo_w[:, 1], fo_w[:, 2], c="purple", s=2, alpha=0.4, marker="s", label="fo_wall")
ax11.scatter(sub(x_d, 15000)[:, 1], sub(x_d, 15000)[:, 2], c="red", s=0.3, alpha=0.5, label="data pts")
ax11.set_xlabel("Y (mm)"); ax11.set_ylabel("Z (mm)")
ax11.set_title("Data points vs wall cells (YZ)")
ax11.legend(fontsize=6, markerscale=3)
ax11.set_aspect("equal")

# 12: Summary text
ax12 = fig.add_subplot(3, 4, 12)
ax12.axis("off")
fi_cell_indices = set(fluid_cls["fluid_i"])
fo_cell_indices = set(fluid_cls["fluid_o"])
summary = (
    f"Sampling Distribution Summary\n"
    f"{'─' * 35}\n"
    f"Mesh zones:\n"
    f"  fluid_i:      {len(fluid_cls['fluid_i']):>10,}\n"
    f"    core:       {len(fluid_cls['fluid_i_core']):>10,}\n"
    f"    wall:       {len(fluid_cls['fluid_i_wall']):>10,}\n"
    f"  fluid_o:      {len(fluid_cls['fluid_o']):>10,}\n"
    f"    core:       {len(fluid_cls['fluid_o_core']):>10,}\n"
    f"    wall:       {len(fluid_cls['fluid_o_wall']):>10,}\n"
    f"  other:        {n_fluid - len(fluid_cls['fluid_i']) - len(fluid_cls['fluid_o']):>10,}\n\n"
    f"Data points ({n_data:,}):\n"
    f"  60% fluid_i   = {int(n_data * 0.60):>8,}\n"
    f"  25% fluid_o   = {int(n_data * 0.25):>8,}\n"
    f"  15% random    = {int(n_data * 0.15):>8,}\n\n"
    f"Colloc points ({n_colloc:,}):\n"
    f"  30% fluid_i   = {int(n_colloc * 0.30):>8,}\n"
    f"  20% all fluid = {int(n_colloc * 0.20):>8,}\n"
    f"  50% all cells = {int(n_colloc * 0.50):>8,}\n\n"
    f"Solid temp ({n_solid_pts:,}):\n"
    f"  80% faces     = {int(n_solid_pts * 0.80):>8,}\n"
    f"  20% cells     = {int(n_solid_pts * 0.20):>8,}\n\n"
    f"All sampling is uniform within each zone\n"
    f"(no wall/core distinction)."
)
ax12.text(0.05, 0.95, summary, transform=ax12.transAxes, fontsize=9, fontfamily="monospace",
          verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

fig.suptitle("PINN Sampling Point Distribution — Uniform within zones (no wall bias)", fontsize=14, y=0.99)
fig.tight_layout()
png_path = OUT_DIR / "sampling_distribution.png"
fig.savefig(png_path, dpi=150)
plt.close(fig)
print(f"\nSaved: {png_path}")

# ── Also write sampled coords to VTK for ParaView ──
def write_vtk_points(filename, x_norm, label_ints=None, subsample=1):
    x_m = x_norm * coord_range + coord_min
    n = len(x_m)
    if subsample > 1:
        idx = np.linspace(0, n-1, n//subsample, dtype=int)
        x_m = x_m[idx]
        if label_ints is not None:
            label_ints = label_ints[idx]
        n = len(x_m)
    with open(OUT_DIR / filename, "w", encoding="utf-8") as f:
        f.write(f"# vtk DataFile Version 3.0\nSampled points ({n:,})\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for p in x_m:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        f.write(f"VERTICES {n} {n * 2}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        if label_ints is not None:
            f.write(f"POINT_DATA {n}\n")
            f.write("SCALARS label int 1\nLOOKUP_TABLE default\n")
            for lbl in label_ints:
                f.write(f"{lbl}\n")

# Write data points and collocation points as VTK
write_vtk_points("sampled_data_points.vtk", x_data.numpy())
print(f"Saved: output/sampled_data_points.vtk")
write_vtk_points("sampled_colloc_points.vtk", x_col.numpy(), subsample=2)
print(f"Saved: output/sampled_colloc_points.vtk")
write_vtk_points("sampled_solid_temp_points.vtk", x_solid.numpy())
print(f"Saved: output/sampled_solid_temp_points.vtk")

print("\nDone.")
