"""Visualize cell zones (soild / fluid_o / fluid_i) from cas.h5.

Use a vectorized owner-only accumulation:
  for each face, add its node coordinates to its owner cell (c0); compute mean.
This gives an approximate cell center suitable for scatter visualization
(cells that only appear as c1 will still get coverage via interior/interface faces
that list them as c0 elsewhere).
"""
import os
import time
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CAS = r"F:\SIG\data\meshData\KMB_phase_change_test.cas.h5"
OUT_DIR = r"F:\SIG\pinn_standalone\output"
OUT_PNG = os.path.join(OUT_DIR, "zones_3d.png")

N_PLOT_PER_ZONE = 25000


def parseNames(raw):
    arr = np.asarray(raw).reshape(-1)
    if arr.size == 0:
        return []
    first = arr[0]
    text = first.decode("utf-8", errors="ignore") if isinstance(first, bytes) else str(first)
    return [p.strip() for p in text.split(";") if p.strip()]


def computeAllCellCenters(nodeCoords, faceC0, faceNn, faceNodes, nCells):
    """Vectorized cell-center estimation by averaging owner-face node positions."""
    ownerPerNode = np.repeat(faceC0 - 1, faceNn)
    nodeIdx = faceNodes - 1
    coords = nodeCoords[nodeIdx]
    sumXyz = np.zeros((nCells, 3), dtype=np.float64)
    cnt = np.zeros(nCells, dtype=np.int64)
    np.add.at(sumXyz, ownerPerNode, coords)
    np.add.at(cnt, ownerPerNode, 1)
    cnt_safe = np.maximum(cnt, 1)
    return sumXyz / cnt_safe[:, None], cnt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Reading cas.h5 ...")
    t0 = time.time()
    with h5py.File(CAS, "r") as f:
        nodeCoords = np.asarray(f["meshes/1/nodes/coords/1"][()])
        faceNn = np.asarray(f["meshes/1/faces/nodes/1/nnodes"][()])
        faceNodes = np.asarray(f["meshes/1/faces/nodes/1/nodes"][()])
        faceC0 = np.asarray(f["meshes/1/faces/c0/1"][()])
        z = f["meshes/1/cells/zoneTopology"]
        zids = np.asarray(z["id"][()]).reshape(-1)
        zmin = np.asarray(z["minId"][()]).reshape(-1)
        zmax = np.asarray(z["maxId"][()]).reshape(-1)
        names = parseNames(np.asarray(z["name"][()]))
    print(f"  done in {time.time()-t0:.1f}s   n_faces={len(faceC0)}  n_nodes={len(nodeCoords)}")
    nCells = int(faceC0.max())
    print(f"  total cells (from c0.max): {nCells}")

    print("Computing cell centers (vectorized) ...")
    t0 = time.time()
    centers, cnt = computeAllCellCenters(nodeCoords, faceC0, faceNn, faceNodes, nCells)
    print(f"  done in {time.time()-t0:.1f}s")
    nMissing = int((cnt == 0).sum())
    print(f"  cells with no owner-face coverage: {nMissing}")

    zoneCells = {}
    for i in range(len(zids)):
        name = names[i] if i < len(names) else f"zone_{zids[i]}"
        zoneCells[name] = (int(zmin[i]), int(zmax[i]))
        cellsLo, cellsHi = zoneCells[name]
        zoneCnt = cnt[cellsLo - 1:cellsHi]
        zoneMissing = int((zoneCnt == 0).sum())
        print(f"  zone {name:<10} cells [{cellsLo}, {cellsHi}]  "
              f"n={cellsHi - cellsLo + 1}  missing={zoneMissing}")

    rng = np.random.default_rng(0)
    sampled = {}
    for name, (lo, hi) in zoneCells.items():
        n = hi - lo + 1
        k = min(N_PLOT_PER_ZONE, n)
        ids = rng.choice(np.arange(lo - 1, hi), size=k, replace=False)
        valid = cnt[ids] > 0
        ids = ids[valid]
        sampled[name] = centers[ids]
        print(f"  plot sample {name}: {len(ids)} of {n}")

    fig = plt.figure(figsize=(14, 12))
    colorMap = {
        "soild":   ("#7f7f7f", "soild (solid wall)"),
        "fluid_o": ("#1f77b4", "fluid_o (outer fluid)"),
        "fluid_i": ("#d62728", "fluid_i (inner fluid)"),
    }

    allC = np.concatenate([c for c in sampled.values() if len(c) > 0], axis=0)
    xLim = (allC[:, 0].min(), allC[:, 0].max())
    yLim = (allC[:, 1].min(), allC[:, 1].max())
    zLim = (allC[:, 2].min(), allC[:, 2].max())

    panels = [
        ("overlay", None, "All zones overlay"),
        ("soild", colorMap["soild"][0], colorMap["soild"][1]),
        ("fluid_o", colorMap["fluid_o"][0], colorMap["fluid_o"][1]),
        ("fluid_i", colorMap["fluid_i"][0], colorMap["fluid_i"][1]),
    ]
    for i, (key, color, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        if key == "overlay":
            order = [("soild", 0.7, 0.6), ("fluid_o", 1.0, 0.20), ("fluid_i", 1.0, 0.35)]
            for name, sScale, alpha in order:
                if name in sampled and len(sampled[name]) > 0:
                    c = sampled[name]
                    ax.scatter(c[:, 0], c[:, 1], c[:, 2],
                               s=1.0 * sScale, c=colorMap[name][0],
                               alpha=alpha, label=colorMap[name][1])
            ax.legend(loc="upper right", fontsize=8, markerscale=6)
        elif key in sampled and len(sampled[key]) > 0:
            c = sampled[key]
            ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=1.0, c=color, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_xlim(xLim); ax.set_ylim(yLim); ax.set_zlim(zLim)
        ax.view_init(elev=22, azim=-58)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)
    print(f"\nSaved {OUT_PNG}")


if __name__ == "__main__":
    main()
