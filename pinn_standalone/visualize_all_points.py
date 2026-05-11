"""Export all mesh points for ParaView + generate PNG preview."""
import sys, h5py
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DataConfig
from data_loader import (
    load_cas, compute_cell_centers, classify_solid_cells, classify_fluid_cells,
    load_multi_fluid_fields, compute_fluid_i_soild_face_data,
)

dc = DataConfig()
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading mesh...")
mesh = load_cas(dc.cas_path)
cell_centers_all = compute_cell_centers(mesh)

print("Finding cell counts...")
dat_files = sorted(dc.dat_dir.glob("*.dat.h5"))
if dat_files:
    _, _, _, ps = load_multi_fluid_fields(dc.dat_dir, "*.dat.h5")
    n_fluid = ps["n_fluid"]
    n_solid = ps["n_solid"]
else:
    n_fluid = 610009
    n_solid = 64503
print(f"  fluid={n_fluid:,}  solid={n_solid:,}  total={n_fluid + n_solid:,}  all_centers={cell_centers_all.shape[0]:,}")

fluid_coords = cell_centers_all[:n_fluid]
solid_coords = cell_centers_all[n_fluid:n_fluid + n_solid]

# Filter out cells with [0,0,0] coords (invalid / no nodes)
solid_valid_mask = (np.abs(solid_coords).max(axis=1) > 1e-10)
n_invalid = (~solid_valid_mask).sum()
if n_invalid > 0:
    invalid_ids = np.where(~solid_valid_mask)[0]
    print(f"  Dropping {n_invalid} solid cells with [0,0,0] coords (ids: {invalid_ids[:10]}...)")
    solid_coords = solid_coords[solid_valid_mask]
    # remap: old_id → new_id (or -1 if dropped)
    old_to_new = np.full(n_solid, -1, dtype=np.int32)
    old_to_new[solid_valid_mask] = np.arange(solid_valid_mask.sum())
    n_solid_eff = solid_valid_mask.sum()
else:
    old_to_new = np.arange(n_solid, dtype=np.int32)
    n_solid_eff = n_solid

# Classification
classification_raw = classify_solid_cells(mesh, n_solid)

# Remap classification to filtered coords
def remap(arr):
    """Filter classification array: drop invalid cells, remap indices."""
    filtered = []
    for cid in arr:
        new_id = old_to_new[cid]
        if new_id >= 0:
            filtered.append(new_id)
    return np.array(filtered, dtype=np.int32)

classification = {}
for key in ["fluid_i_soild", "fluid_o_soild", "soild_boundary", "interior"]:
    classification[key] = remap(classification_raw[key])

# Use filtered n_solid for face data
coord_min = fluid_coords.min(axis=0)
coord_max = fluid_coords.max(axis=0)
coord_range = coord_max - coord_min + 1e-12

face_data = compute_fluid_i_soild_face_data(mesh, n_solid_eff, classification, coord_min, coord_range)

# Remap face→cell since some cells might have been dropped
face_to_cell_raw = face_data["face_to_cell"]
face_to_cell_remapped = np.array([old_to_new[c] if 0 <= c < n_solid else -1 for c in face_to_cell_raw], dtype=np.int32)

# ── 1. Fluid cell centers (uniform stride subsample) ──
ss = max(1, len(fluid_coords) // 100000)
fluid_viz = fluid_coords[::ss]
# Labels: 0=other, 1=fluid_i_core, 2=fluid_i_wall, 3=fluid_o_core, 4=fluid_o_wall
fluid_cls = classify_fluid_cells(mesh, n_fluid)
fluid_labels = np.zeros(n_fluid, dtype=np.int32)
fluid_labels[fluid_cls["fluid_o_core"]] = 3
fluid_labels[fluid_cls["fluid_o_wall"]] = 4
fluid_labels[fluid_cls["fluid_i_core"]] = 1
fluid_labels[fluid_cls["fluid_i_wall"]] = 2  # highest priority (wall overlay)

vtk_fluid = OUT_DIR / "all_fluid_cells.vtk"
with open(vtk_fluid, "w", encoding="utf-8") as f:
    n = len(fluid_viz)
    f.write(f"# vtk DataFile Version 3.0\nFluid cell centers ({n:,} of {n_fluid:,}, stride={ss})\nASCII\nDATASET POLYDATA\n")
    f.write(f"POINTS {n} float\n")
    for p in fluid_viz:
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    f.write(f"VERTICES {n} {n * 2}\n")
    for i in range(n):
        f.write(f"1 {i}\n")
    f.write(f"POINT_DATA {n}\n")
    f.write("SCALARS zone int 1\nLOOKUP_TABLE default\n")
    for lbl in fluid_labels[::ss][:n]:
        f.write(f"{lbl}\n")
print(f"Saved: {vtk_fluid}")

# ── 2. Solid cell centers with classification ──
# Label order: fluid_i_soild=1, fluid_o_soild=2, soild_boundary=3, interior=0
solid_labels = np.zeros(n_solid_eff, dtype=np.int32)
for cid in classification["fluid_i_soild"]:
    solid_labels[cid] = 1
for cid in classification["fluid_o_soild"]:
    if solid_labels[cid] == 0:  # don't overwrite fi (corner cells)
        solid_labels[cid] = 2
for cid in classification["soild_boundary"]:
    if solid_labels[cid] == 0:
        solid_labels[cid] = 3

vtk_solid = OUT_DIR / "all_solid_cells.vtk"
with open(vtk_solid, "w", encoding="utf-8") as f:
    n = n_solid_eff
    f.write(f"# vtk DataFile Version 3.0\nSolid cell centers ({n:,})\nASCII\nDATASET POLYDATA\n")
    f.write(f"POINTS {n} float\n")
    for p in solid_coords:
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    f.write(f"VERTICES {n} {n * 2}\n")
    for i in range(n):
        f.write(f"1 {i}\n")
    f.write(f"POINT_DATA {n}\n")
    f.write("SCALARS classification int 1\nLOOKUP_TABLE default\n")
    for lbl in solid_labels:
        f.write(f"{lbl}\n")
print(f"Saved: {vtk_solid}")

# ── 3. Face centers (interface) ──
face_coords_m = face_data["face_centers_m"]
vtk_faces_all = OUT_DIR / "all_fluid_i_soild_faces.vtk"
with open(vtk_faces_all, "w", encoding="utf-8") as f:
    n = len(face_coords_m)
    f.write(f"# vtk DataFile Version 3.0\nfluid_i-soild face centers ({n:,})\nASCII\nDATASET POLYDATA\n")
    f.write(f"POINTS {n} float\n")
    for p in face_coords_m:
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    f.write(f"VERTICES {n} {n * 2}\n")
    for i in range(n):
        f.write(f"1 {i}\n")
print(f"Saved: {vtk_faces_all}")

# ── 4. Generate PNG preview ──
fig = plt.figure(figsize=(20, 12))

# Subsample for plotting
fs_plot = max(1, len(face_coords_m) // 8000)
cs_plot = max(1, n_solid_eff // 8000)

# 4a: 3D overview — solid cells colored by classification + faces
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
colors = {0: "gray", 1: "red", 2: "blue", 3: "green"}
for lbl, name, marker, size in [(1, "fluid_i_soild", "o", 2), (2, "fluid_o_soild", "s", 2), (3, "soild_boundary", "^", 4), (0, "interior", ".", 1)]:
    mask = solid_labels[::cs_plot] == lbl
    if mask.any():
        ax1.scatter(solid_coords[::cs_plot][mask, 0], solid_coords[::cs_plot][mask, 1], solid_coords[::cs_plot][mask, 2],
                    c=colors[lbl], s=size, alpha=0.6, marker=marker, label=f"{name} ({mask.sum():,})")
# Faces in cyan
ax1.scatter(face_coords_m[::fs_plot, 0], face_coords_m[::fs_plot, 1], face_coords_m[::fs_plot, 2],
            c="cyan", s=1, alpha=0.9, label=f"faces ({len(face_coords_m):,})")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
ax1.set_title("Solid cells (colored) + Interface faces (cyan)")
ax1.legend(fontsize=6, loc="upper right")
ax1.view_init(elev=30, azim=-60)

# 4b: Top view
ax2 = fig.add_subplot(2, 3, 2, projection="3d")
for lbl, name, marker, size in [(1, "fluid_i_soild", "o", 2), (2, "fluid_o_soild", "s", 2)]:
    mask = solid_labels[::cs_plot] == lbl
    if mask.any():
        ax2.scatter(solid_coords[::cs_plot][mask, 0], solid_coords[::cs_plot][mask, 1], solid_coords[::cs_plot][mask, 2],
                    c=colors[lbl], s=size, alpha=0.6, marker=marker)
ax2.scatter(face_coords_m[::fs_plot, 0], face_coords_m[::fs_plot, 1], face_coords_m[::fs_plot, 2],
            c="cyan", s=1, alpha=0.9)
ax2.view_init(elev=90, azim=-90)
ax2.set_title("Top view (XZ) — fi=red, fo=blue, faces=cyan")

# 4c: Y-Z projection of solid cells
ax3 = fig.add_subplot(2, 3, 3)
for lbl, name, marker, size in [(1, "fluid_i_soild", "o", 1), (0, "interior", ".", 0.5)]:
    mask = solid_labels[::cs_plot] == lbl
    if mask.any():
        ax3.scatter(solid_coords[::cs_plot][mask, 1], solid_coords[::cs_plot][mask, 2],
                    c=colors[lbl], s=size, alpha=0.5, marker=marker)
ax3.scatter(face_coords_m[::fs_plot, 1], face_coords_m[::fs_plot, 2],
            c="cyan", s=1, alpha=0.8)
ax3.set_xlabel("Y (m)"); ax3.set_ylabel("Z (m)")
ax3.set_title("Y-Z projection — fi (red) + faces (cyan)")
ax3.set_aspect("equal"); ax3.grid(True, alpha=0.3)

# 4d: Classification pie chart
ax4 = fig.add_subplot(2, 3, 4)
counts = [len(classification["fluid_i_soild"]), len(classification["fluid_o_soild"]),
          len(classification["soild_boundary"]), len(classification["interior"])]
names = [f"fluid_i_soild\n{counts[0]:,}", f"fluid_o_soild\n{counts[1]:,}",
         f"soild_boundary\n{counts[2]:,}", f"interior\n{counts[3]:,}"]
pie_colors = ["red", "blue", "green", "gray"]
ax4.pie(counts, labels=names, colors=pie_colors, autopct="%1.1f%%", startangle=90)
ax4.set_title(f"Solid cell classification ({n_solid_eff:,} total)")

# 4e: Face→cell distance distribution
from data_loader import load_cas
face_to_cell_dist = np.linalg.norm(face_coords_m - solid_coords[face_to_cell_remapped], axis=1) * 1000
ax5 = fig.add_subplot(2, 3, 5)
ax5.hist(face_to_cell_dist, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
ax5.axvline(face_to_cell_dist.mean(), color="red", linestyle="--", label=f"Mean={face_to_cell_dist.mean():.1f} mm")
ax5.axvline(np.median(face_to_cell_dist), color="orange", linestyle="--", label=f"Median={np.median(face_to_cell_dist):.1f} mm")
ax5.set_xlabel("Face → cell distance (mm)"); ax5.set_ylabel("Count")
ax5.set_title("Face→cell center distance")
ax5.legend(); ax5.grid(True, alpha=0.3)

# 4f: Summary text
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis("off")
summary = (
    f"Mesh Summary\n"
    f"{'─' * 30}\n"
    f"Fluid cells:      {n_fluid:>10,}\n"
    f"Solid cells:      {n_solid:>10,}\n"
    f"  (dropped [0,0,0]): {n_invalid:>6,}\n"
    f"  fluid_i_soild:  {counts[0]:>10,}\n"
    f"  fluid_o_soild:  {counts[1]:>10,}\n"
    f"  soild_boundary: {counts[2]:>10,}\n"
    f"  interior:       {counts[3]:>10,}\n"
    f"Face centers:     {len(face_coords_m):>10,}\n"
    f"Face→cell dist:   mean={face_to_cell_dist.mean():.1f} mm\n"
    f"                   median={np.median(face_to_cell_dist):.1f} mm\n\n"
    f"PINN solid temp sampling:\n"
    f"  70% from face centers (interface)\n"
    f"  20% from fluid_o_soild cells\n"
    f"  10% from random solid cells"
)
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=11, fontfamily="monospace",
         verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

fig.suptitle("PINN Mesh Overview — All Points", fontsize=14, y=0.99)
fig.tight_layout()
png_path = OUT_DIR / "all_points_overview.png"
fig.savefig(png_path, dpi=150)
plt.close(fig)
print(f"Saved: {png_path}")

print(f"\n=== Summary ===")
print(f"  Fluid: {n_fluid:,} | Solid: {n_solid:,} (valid: {n_solid_eff:,}, dropped [0,0,0]: {n_invalid})")
print(f"  fi={counts[0]:,}  fo={counts[1]:,}  boundary={counts[2]:,}  interior={counts[3]:,}")
print(f"  Faces: {len(face_coords_m):,}")
print("Done.")
