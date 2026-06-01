"""
Visualize the full solid domain mesh from cas.h5.
Saves a 3D scatter plot + VTK to output dir.
"""
import h5py, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_cas, compute_cell_centers

DATA_DIR = Path(__file__).parent.parent / "data"
CAS_PATH = DATA_DIR / "meshData/KMB_phase_change_test.cas.h5"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading mesh...")
mesh = load_cas(CAS_PATH)
cell_centers = compute_cell_centers(mesh)
print(f"Total cells: {cell_centers.shape[0]}")

# Solid cells: determined from data loading (n_total - n_fluid)
from data_loader import load_multi_fluid_fields
dat_files = sorted((DATA_DIR / "T_160_200").glob("*.dat.h5"))
if dat_files:
    _, _, _, ps = load_multi_fluid_fields(DATA_DIR / "T_160_200", "*.dat.h5")
    n_fluid = ps["n_fluid"]
    n_solid = ps["n_solid"]
else:
    n_fluid = 610009
    n_solid = 64503
fluid_coords = cell_centers[n_solid + 1 : n_solid + 1 + n_fluid]
solid_coords = cell_centers[1 : n_solid + 1]
print(f"Fluid cells: {n_fluid}, Solid cells: {n_solid}")

# Stats
for i, axis in enumerate(["X", "Y", "Z"]):
    print(f"  {axis}: [{solid_coords[:, i].min():.5f}, {solid_coords[:, i].max():.5f}]  "
          f"span={solid_coords[:, i].max() - solid_coords[:, i].min():.5f}")

# ===== 3D scatter plot =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(18, 6))

for view_idx, (elev, azim, title) in enumerate([
    (30, -60, "Solid domain — isometric"),
    (90, -90, "Solid domain — top (XZ)"),
    (0, 0, "Solid domain — front (XY)"),
]):
    ax = fig.add_subplot(1, 3, view_idx + 1, projection="3d")
    # Subsample if too many cells for rendering
    n_plot = min(len(solid_coords), 20000)
    idx = np.linspace(0, len(solid_coords) - 1, n_plot, dtype=int)
    coords_plot = solid_coords[idx]
    ax.scatter(coords_plot[:, 0], coords_plot[:, 1], coords_plot[:, 2],
               c=coords_plot[:, 1], s=0.5, alpha=0.7, cmap="viridis")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

fig.suptitle(f"Solid Domain — {solid_coords.shape[0]:,} cells", fontsize=14)
fig.tight_layout()
png_path = OUT_DIR / "solid_domain_mesh.png"
fig.savefig(png_path, dpi=200)
plt.close(fig)
print(f"Saved: {png_path}")

# ===== VTK for ParaView =====
vtk_path = OUT_DIR / "solid_domain_mesh.vtk"
with open(vtk_path, "w", encoding="utf-8") as f:
    n = solid_coords.shape[0]
    f.write("# vtk DataFile Version 3.0\nsolid domain mesh\nASCII\nDATASET POLYDATA\n")
    f.write(f"POINTS {n} float\n")
    for p in solid_coords:
        f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    f.write(f"VERTICES {n} {n * 2}\n")
    for i in range(n):
        f.write(f"1 {i}\n")
    f.write(f"POINT_DATA {n}\n")
    f.write("SCALARS cell_index int 1\nLOOKUP_TABLE default\n")
    for i in range(n):
        f.write(f"{i}\n")
print(f"Saved: {vtk_path}")

# ===== Also visualize fluid domain boundary =====
# Sample fluid cells near the solid interface
# fluid_coords already defined above using correct cell zone split
# Find fluid cells closest to solid (Y > some threshold)
y_threshold = solid_coords[:, 1].min()
nearby_fluid = fluid_coords[fluid_coords[:, 1] >= y_threshold - 0.002]

fig2 = plt.figure(figsize=(12, 10))
ax2 = fig2.add_subplot(111, projection="3d")
# Solid in red
n_s_plot = min(len(solid_coords), 15000)
idx_s = np.linspace(0, len(solid_coords) - 1, n_s_plot, dtype=int)
ax2.scatter(solid_coords[idx_s, 0], solid_coords[idx_s, 1], solid_coords[idx_s, 2],
            c="red", s=0.5, alpha=0.6, label="Solid")
# Nearby fluid in blue
n_f_plot = min(len(nearby_fluid), 15000)
idx_f = np.linspace(0, len(nearby_fluid) - 1, n_f_plot, dtype=int)
ax2.scatter(nearby_fluid[idx_f, 0], nearby_fluid[idx_f, 1], nearby_fluid[idx_f, 2],
            c="blue", s=0.3, alpha=0.4, label="Fluid (near solid)")
ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_zlabel("Z (m)")
ax2.set_title("Solid (red) + nearby fluid (blue)")
ax2.legend()
ax2.view_init(elev=30, azim=-60)
fig2.tight_layout()
png_path2 = OUT_DIR / "solid_fluid_interface.png"
fig2.savefig(png_path2, dpi=200)
plt.close(fig2)
print(f"Saved: {png_path2}")

print("\nDone. Open the PNG files or the VTK in ParaView.")
