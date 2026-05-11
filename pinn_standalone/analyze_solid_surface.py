"""
Identify and visualize solid inner-surface cells (fluid_i-soild interface).
These are the cells whose temperature matters for sterilization.
"""
import h5py, numpy as np
from pathlib import Path

CAS_PATH = Path(r"F:\SIG\data\meshData\KMB_phase_change_test.cas.h5")
DAT_DIR = Path(r"F:\SIG\data\T_160_200")
OUT_DIR = Path(r"F:\SIG\pinn_standalone\output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading mesh...")
with h5py.File(CAS_PATH, "r") as f:
    node_coords = np.asarray(f["meshes/1/nodes/coords/1"][()])
    face_nn = np.asarray(f["meshes/1/faces/nodes/1/nnodes"][()])
    face_nodes = np.asarray(f["meshes/1/faces/nodes/1/nodes"][()])
    face_c0 = np.asarray(f["meshes/1/faces/c0/1"][()])
    offsets = np.cumsum(np.concatenate([[0], face_nn]))

    fz = f["meshes/1/faces/zoneTopology"]
    fz_mins = np.asarray(fz["minId"][()]).reshape(-1)
    fz_maxs = np.asarray(fz["maxId"][()]).reshape(-1)

# Zone face ranges (1-based face IDs)
# fluid_i-soild = zone 41: face_min=?, face_max=?
# Look up zone 41 from the face zones
# We need the face IDs that belong to fluid_i-soild
# From earlier: face zone ID=41 has face_min from fz_mins[?] to fz_maxs[?]

# Actually need to find the right zone index. From earlier output:
# ID=41: "fluid_i-soild" faces=30175
# But we need the index in the fz arrays. Let me re-read.

with h5py.File(CAS_PATH, "r") as f:
    fz = f["meshes/1/faces/zoneTopology"]
    fz_ids = np.asarray(fz["id"][()]).reshape(-1)
    fz_mins = np.asarray(fz["minId"][()]).reshape(-1)
    fz_maxs = np.asarray(fz["maxId"][()]).reshape(-1)
    fz_c0 = np.asarray(fz["c0"][()]).reshape(-1)

# Find fluid_i-soild zone (id=41)
for i in range(len(fz_ids)):
    if fz_ids[i] == 41:
        inner_solid_faces_min = fz_mins[i]  # 1-based
        inner_solid_faces_max = fz_maxs[i]
        print(f"fluid_i-soild: faces [{inner_solid_faces_min}, {inner_solid_faces_max}] = {inner_solid_faces_max - inner_solid_faces_min + 1} faces")
        break

# Get the cell IDs (c0) for these faces → these are the fluid_i cells adjacent to solid
# But we want the solid cells, which are c1 for the shadow faces.
# Actually: fluid_i-soild has c0=167 (fluid_i), c1=164 (soild)
# Shadow zone (id=3) has c0=164 (soild), c1=167 (fluid_i)
# Let's use the shadow zone to get solid-side cells

for i in range(len(fz_ids)):
    if fz_ids[i] == 3:
        shadow_face_min = fz_mins[i]
        shadow_face_max = fz_maxs[i]
        print(f"fluid_i-soild-shadow: faces [{shadow_face_min}, {shadow_face_max}]")
        break

# The shadow zone gives us faces FROM solid TO fluid_i
# So face_c0 for these faces = solid cell ID

# Get solid cells adjacent to inner surface
inner_surface_solid_cells = set()
for fid in range(shadow_face_min - 1, shadow_face_max):
    c0 = int(face_c0[fid])
    if c0 >= 0:
        inner_surface_solid_cells.add(c0)

print(f"Inner surface solid cells (adjacent to fluid_i): {len(inner_surface_solid_cells)}")

# Now categorize all 64503 solid cells
solid_ids = set(range(64503))  # 0-based
outer_surface_cells = set()
inner_surface_cells = set()

# fluid_o-soild-shadow zone (id=2) → solid cells adjacent to outer fluid
for i in range(len(fz_ids)):
    if fz_ids[i] == 2:
        for fid in range(fz_mins[i] - 1, fz_maxs[i]):
            c0 = int(face_c0[fid])
            if c0 >= 0 and c0 < 64503:
                outer_surface_cells.add(c0)

# soild:1 zone (id=40) → solid cells on exterior boundary
exterior_cells = set()
for i in range(len(fz_ids)):
    if fz_ids[i] == 40:
        for fid in range(fz_mins[i] - 1, fz_maxs[i]):
            c0 = int(face_c0[fid])
            if c0 >= 0 and c0 < 64503:
                exterior_cells.add(c0)

# fluid_i-soild-shadow (id=3) → solid cells adjacent to inner fluid
for i in range(len(fz_ids)):
    if fz_ids[i] == 3:
        for fid in range(fz_mins[i] - 1, fz_maxs[i]):
            c0 = int(face_c0[fid])
            if c0 >= 0 and c0 < 64503:
                inner_surface_cells.add(c0)

# Interior solid cells = all solid - all surface cells
all_surface = outer_surface_cells | inner_surface_cells | exterior_cells
interior_solid = solid_ids - all_surface

print(f"\n=== Solid cell classification ===")
print(f"  Inner surface (contact fluid_i): {len(inner_surface_cells):,}")
print(f"  Outer surface (contact fluid_o): {len(outer_surface_cells):,}")
print(f"  Exterior boundary (soild:1):     {len(exterior_cells):,}")
print(f"  Interior (bulk solid):           {len(interior_solid):,}")
print(f"  Total solid cells:               {len(solid_ids):,}")

# Compute cell centers for solid cells
print("\nComputing cell centers for solid cells...")
# Reuse mesh loading
with h5py.File(CAS_PATH, "r") as f:
    node_coords = np.asarray(f["meshes/1/nodes/coords/1"][()])
    face_nn = np.asarray(f["meshes/1/faces/nodes/1/nnodes"][()])
    face_nodes = np.asarray(f["meshes/1/faces/nodes/1/nodes"][()])
    face_c0_raw = np.asarray(f["meshes/1/faces/c0/1"][()])
    offsets = np.cumsum(np.concatenate([[0], face_nn]))

# Only compute for solid cells (0 to 64502)
n_solid = 64503
cell_node_sets = [set() for _ in range(n_solid)]
n_faces = len(face_nn)

for fid in range(n_faces):
    owner = int(face_c0_raw[fid])
    if owner < 0 or owner >= n_solid:
        continue
    for nid in face_nodes[offsets[fid]:offsets[fid + 1]]:
        cell_node_sets[owner].add(int(nid) - 1)

solid_centers = np.zeros((n_solid, 3), dtype=np.float64)
for cid in range(n_solid):
    nids = np.array(list(cell_node_sets[cid]))
    if len(nids) > 0:
        solid_centers[cid] = node_coords[nids].mean(axis=0)

# Stats for inner surface
inner_list = sorted(inner_surface_cells)
inner_coords = solid_centers[inner_list]
print(f"Inner surface coordinates:")
print(f"  X: [{inner_coords[:,0].min():.5f}, {inner_coords[:,0].max():.5f}]")
print(f"  Y: [{inner_coords[:,1].min():.5f}, {inner_coords[:,1].max():.5f}]")
print(f"  Z: [{inner_coords[:,2].min():.5f}, {inner_coords[:,2].max():.5f}]")

# ===== Visualization =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 12))

views = [
    (30, -60, "Isometric view"),
    (90, -90, "Top view (XZ plane)"),
    (0, 0, "Front view (XY plane)"),
]

for vi, (elev, azim, title) in enumerate(views):
    ax = fig.add_subplot(2, 3, vi + 1, projection="3d")

    # Interior in light gray (subsampled)
    if interior_solid:
        il = sorted(interior_solid)
        n_int = min(len(il), 5000)
        idx = np.linspace(0, len(il) - 1, n_int, dtype=int)
        ax.scatter(solid_centers[il][idx, 0], solid_centers[il][idx, 1], solid_centers[il][idx, 2],
                   c="lightgray", s=0.3, alpha=0.3, label=f"Interior ({len(interior_solid):,})")

    # Outer surface in blue
    if outer_surface_cells:
        ol = sorted(outer_surface_cells)
        ax.scatter(solid_centers[ol, 0], solid_centers[ol, 1], solid_centers[ol, 2],
                   c="steelblue", s=1, alpha=0.6, label=f"Outer surf ({len(outer_surface_cells):,})")

    # Inner surface in RED (the important one!)
    if inner_surface_cells:
        il2 = sorted(inner_surface_cells)
        ax.scatter(solid_centers[il2, 0], solid_centers[il2, 1], solid_centers[il2, 2],
                   c="red", s=3, alpha=0.9, label=f"Inner surf ({len(inner_surface_cells):,})")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(markerscale=5, fontsize=8)
    ax.view_init(elev=elev, azim=azim)

# Bottom row: temperature on inner surface at latest timestep
# Load temperature data
print("\nLoading solid temperature data...")
dat_files = sorted(DAT_DIR.glob("*.dat.h5"))
if dat_files:
    with h5py.File(dat_files[-1], "r") as f:
        T_solid = np.asarray(f["results/1/phase-1/cells/SV_T/1"][()])
    print(f"  t={dat_files[-1].stem}, T range: [{T_solid.min():.2f}, {T_solid.max():.2f}] K")

    T_inner = T_solid[inner_list]
    T_outer = T_solid[sorted(outer_surface_cells)] if outer_surface_cells else np.array([])

    # Inner surface temperature map
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    sc4 = ax4.scatter(inner_coords[:, 0], inner_coords[:, 1], inner_coords[:, 2],
                      c=T_inner, s=3, cmap="hot", alpha=0.9)
    ax4.set_xlabel("X (m)"); ax4.set_ylabel("Y (m)"); ax4.set_zlabel("Z (m)")
    ax4.set_title(f"Inner surface T (K) — mean={T_inner.mean():.1f}K")
    ax4.view_init(elev=30, azim=-60)
    plt.colorbar(sc4, ax=ax4, shrink=0.6)

    # Cross-section: T vs Y coordinate
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(inner_coords[:, 1], T_inner, s=2, alpha=0.5, c="red", label=f"Inner (n={len(T_inner):,})")
    if len(T_outer) > 0:
        outer_coords = solid_centers[sorted(outer_surface_cells)]
        ax5.scatter(outer_coords[:, 1], T_outer, s=1, alpha=0.3, c="steelblue", label=f"Outer (n={len(T_outer):,})")
    ax5.set_xlabel("Y coordinate (m)"); ax5.set_ylabel("Temperature (K)")
    ax5.set_title("T vs Y (inner surface = red)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Histogram
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(T_inner, bins=50, color="red", alpha=0.7, label=f"Inner surface")
    if len(T_outer) > 0:
        ax6.hist(T_outer, bins=50, color="steelblue", alpha=0.5, label=f"Outer surface")
    ax6.set_xlabel("Temperature (K)"); ax6.set_ylabel("Count")
    ax6.set_title(f"T distribution (latest timestep)")
    ax6.legend()

fig.suptitle(f"Solid Domain Analysis — Inner Surface ({len(inner_surface_cells):,} cells) is the prediction target",
             fontsize=14)
fig.tight_layout()
png_path = OUT_DIR / "solid_surface_analysis.png"
fig.savefig(png_path, dpi=200)
plt.close(fig)
print(f"\nSaved: {png_path}")

# ===== Save cell classification to file =====
cells_path = OUT_DIR / "solid_cell_classification.npz"
np.savez(cells_path,
         inner_surface=np.array(sorted(inner_surface_cells), dtype=np.int32),
         outer_surface=np.array(sorted(outer_surface_cells), dtype=np.int32),
         exterior=np.array(sorted(exterior_cells), dtype=np.int32),
         interior=np.array(sorted(interior_solid), dtype=np.int32),
         solid_centers=solid_centers)
print(f"Saved: {cells_path}")
print("\nDone!")
