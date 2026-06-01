"""Visualize fluid-solid interface — combined fluid_i-soild + fluid_o-soild face centers."""
import sys, h5py
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DataConfig
from data_loader import (
    load_cas, compute_cell_centers, classify_solid_cells,
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
print(f"  n_fluid={n_fluid:,}, n_solid={n_solid:,}")

print("Classifying...")
classification = classify_solid_cells(mesh, n_solid)
for key in ["fluid_i_soild", "fluid_o_soild", "soild_boundary", "interior"]:
    print(f"  {key}: {len(classification[key]):,}")

solid_centers_all = cell_centers_all[1:n_solid + 1]

# Filter invalid [0,0,0] cells
solid_valid = (np.abs(solid_centers_all).max(axis=1) > 1e-10)
n_invalid = (~solid_valid).sum()
if n_invalid:
    print(f"  Dropping {n_invalid} invalid [0,0,0] cells")
    old_to_new = np.full(n_solid, -1, dtype=np.int32)
    old_to_new[solid_valid] = np.arange(solid_valid.sum())
else:
    old_to_new = np.arange(n_solid, dtype=np.int32)

# Compute combined face data (fluid_i + fluid_o shadow zones)
coord_min = cell_centers_all[n_solid + 1 : n_solid + 1 + n_fluid].min(axis=0)
coord_max = cell_centers_all[n_solid + 1 : n_solid + 1 + n_fluid].max(axis=0)
coord_range = coord_max - coord_min + 1e-12

face_data = compute_fluid_i_soild_face_data(mesh, n_solid, classification, coord_min, coord_range)
face_centers = face_data["face_centers_m"]
face_to_cell = face_data["face_to_cell"]
n_faces = len(face_centers)
print(f"\nCombined interface faces: {n_faces:,}")

# Remap face→cell for invalid cells
face_to_cell_remapped = np.array([old_to_new[c] if 0 <= c < n_solid else -1 for c in face_to_cell], dtype=np.int32)

# Face→owner-cell distances
cell_centers_per_face = solid_centers_all[face_to_cell]
face_to_cell_dist = np.linalg.norm(face_centers - cell_centers_per_face, axis=1)
dist_mm = face_to_cell_dist * 1000

print(f"\nFace → cell center distance:")
print(f"  mean:   {dist_mm.mean():.1f} mm")
print(f"  median: {np.median(dist_mm):.1f} mm")
print(f"  min:    {dist_mm.min():.1f} mm")
print(f"  max:    {dist_mm.max():.1f} mm")
print(f"  < 5mm:  {(dist_mm < 5).sum()} / {n_faces} ({(dist_mm < 5).sum()/n_faces*100:.1f}%)")
print(f"  5-30mm: {((dist_mm >= 5) & (dist_mm < 30)).sum()} / {n_faces}")
print(f"  30-80mm:{((dist_mm >= 30) & (dist_mm < 80)).sum()} / {n_faces}")
print(f"  > 80mm: {(dist_mm >= 80).sum()} / {n_faces}")

# ── VTK exports ──
# Face centers with distance scalar
vtk_faces = OUT_DIR / "fluid_i_soild_faces.vtk"
with open(vtk_faces, "w", encoding="utf-8") as f:
    f.write(f"# vtk DataFile Version 3.0\nComplete fluid-solid interface ({n_faces:,} face centers)\nASCII\nDATASET POLYDATA\n")
    f.write(f"POINTS {n_faces} float\n")
    for p in face_centers:
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    f.write(f"VERTICES {n_faces} {n_faces * 2}\n")
    for i in range(n_faces):
        f.write(f"1 {i}\n")
    f.write(f"POINT_DATA {n_faces}\n")
    f.write("SCALARS distance_to_cell_mm float 1\nLOOKUP_TABLE default\n")
    for d in dist_mm:
        f.write(f"{d:.3f}\n")
    f.write("SCALARS owner_cell_id int 1\nLOOKUP_TABLE default\n")
    for cid in face_to_cell:
        f.write(f"{cid}\n")
print(f"Saved: {vtk_faces}")

# Same data as "cells" (PINN prediction points)
vtk_cells = OUT_DIR / "fluid_i_soild_cells.vtk"
with open(vtk_cells, "w", encoding="utf-8") as f:
    f.write(f"# vtk DataFile Version 3.0\nPINN prediction points ({n_faces:,} face centers)\nASCII\nDATASET POLYDATA\n")
    f.write(f"POINTS {n_faces} float\n")
    for p in face_centers:
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    f.write(f"VERTICES {n_faces} {n_faces * 2}\n")
    for i in range(n_faces):
        f.write(f"1 {i}\n")
    f.write(f"POINT_DATA {n_faces}\n")
    f.write("SCALARS owner_cell_id int 1\nLOOKUP_TABLE default\n")
    for cid in face_to_cell:
        f.write(f"{cid}\n")
print(f"Saved: {vtk_cells}")

# ── PNG: 6-panel overview ──
fig = plt.figure(figsize=(22, 15))
fs = max(1, n_faces // 15000)

# 1: 3D interface colored by distance
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
sc1 = ax1.scatter(face_centers[::fs, 0], face_centers[::fs, 1], face_centers[::fs, 2],
                  c=dist_mm[::fs], s=4, cmap="RdYlGn_r", alpha=0.9)
ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
ax1.set_title(f"Full interface ({n_faces:,} faces)\ncolored by face→cell distance (mm)")
ax1.view_init(elev=30, azim=-60)
fig.colorbar(sc1, ax=ax1, shrink=0.5, label="mm")

# 2: Top view
ax2 = fig.add_subplot(2, 3, 2, projection="3d")
ax2.scatter(face_centers[::fs, 0], face_centers[::fs, 1], face_centers[::fs, 2],
            c=dist_mm[::fs], s=4, cmap="RdYlGn_r", alpha=0.9)
ax2.view_init(elev=90, azim=-90)
ax2.set_title("Top view (XZ)")

# 3: Y-Z projection
ax3 = fig.add_subplot(2, 3, 3)
ax3.scatter(face_centers[::fs, 1], face_centers[::fs, 2],
            c="steelblue", s=1, alpha=0.6)
ax3.set_xlabel("Y (m)"); ax3.set_ylabel("Z (m)")
ax3.set_title(f"Y-Z projection — {n_faces:,} faces")
ax3.set_aspect("equal"); ax3.grid(True, alpha=0.3)

# 4: Distance histogram
ax4 = fig.add_subplot(2, 3, 4)
ax4.hist(dist_mm, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
ax4.axvline(dist_mm.mean(), color="red", linestyle="--", label=f"Mean={dist_mm.mean():.1f} mm")
ax4.axvline(np.median(dist_mm), color="orange", linestyle="--", label=f"Median={np.median(dist_mm):.1f} mm")
ax4.set_xlabel("Face → cell distance (mm)"); ax4.set_ylabel("Count")
ax4.set_title("Distance distribution")
ax4.legend(); ax4.grid(True, alpha=0.3)

# 5: Y-Z slice with face→cell arrows
x_mid = face_centers[:, 0].mean()
mask = np.abs(face_centers[:, 0] - x_mid) < 0.005
ax5 = fig.add_subplot(2, 3, 5)
if mask.sum() > 0:
    idx_slice = np.where(mask)[0]
    n_arrows = min(100, len(idx_slice))
    arrow_idx = np.random.default_rng(42).choice(idx_slice, size=n_arrows, replace=False)
    for i in arrow_idx:
        fy, fz = face_centers[i, 1], face_centers[i, 2]
        cy, cz = cell_centers_per_face[i, 1], cell_centers_per_face[i, 2]
        ax5.arrow(fy, fz, cy - fy, cz - fz, head_width=0.001, head_length=0.002,
                  fc="gray", ec="gray", alpha=0.4, linewidth=0.5)
    ax5.scatter(face_centers[idx_slice, 1], face_centers[idx_slice, 2],
                c="red", s=1, alpha=0.5, label="Faces (on interface)")
    ax5.scatter(cell_centers_per_face[idx_slice, 1], cell_centers_per_face[idx_slice, 2],
                c="blue", s=3, alpha=0.7, marker="^", label="Owner cells")
ax5.set_xlabel("Y (m)"); ax5.set_ylabel("Z (m)")
ax5.set_title(f"Y-Z slice at X≈{x_mid*1000:.0f}mm — arrows: face→cell")
ax5.legend(fontsize=7); ax5.set_aspect("equal"); ax5.grid(True, alpha=0.3)

# 6: Cumulative distribution
ax6 = fig.add_subplot(2, 3, 6)
sorted_idx = np.argsort(dist_mm)
ax6.plot(dist_mm[sorted_idx], np.arange(n_faces), linewidth=1, color="steelblue")
ax6.set_xlabel("Face → cell distance (mm)"); ax6.set_ylabel("Face index (sorted)")
ax6.set_title("Cumulative distribution")
ax6.grid(True, alpha=0.3)

fig.suptitle(
    f"Complete fluid-solid Interface — {n_faces:,} faces from fluid_i-soild + fluid_o-soild shadow zones\n"
    f"Cell offset: mean={dist_mm.mean():.1f}mm median={np.median(dist_mm):.1f}mm range=[{dist_mm.min():.0f},{dist_mm.max():.0f}]mm",
    fontsize=12, y=0.99)
fig.tight_layout()

png_path = OUT_DIR / "fluid_i_soild_overlay.png"
fig.savefig(png_path, dpi=180)
plt.close(fig)
print(f"\nSaved: {png_path}")
print("Done.")
