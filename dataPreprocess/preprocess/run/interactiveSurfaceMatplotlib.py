from pathlib import Path

import h5py
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parseZoneNames(rawNames: np.ndarray) -> list[str]:
    if rawNames.size == 0:
        return []
    first = rawNames.reshape(-1)[0]
    if isinstance(first, bytes):
        text = first.decode("utf-8", errors="ignore")
    else:
        text = str(first)
    return [part.strip() for part in text.split(";") if part.strip()]


def faceOffsets(nnodes: np.ndarray) -> np.ndarray:
    n = np.asarray(nnodes, dtype=np.int64).reshape(-1)
    offsets = np.zeros(n.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(n)
    return offsets


def collectBoundaryFaceIndices(zoneTopo: dict[str, np.ndarray]) -> np.ndarray:
    minIds = np.asarray(zoneTopo["minId"]).reshape(-1)
    maxIds = np.asarray(zoneTopo["maxId"]).reshape(-1)
    c1 = np.asarray(zoneTopo["c1"]).reshape(-1)
    names = parseZoneNames(np.asarray(zoneTopo["name"]))
    selected: list[np.ndarray] = []
    for i in range(minIds.size):
        if int(c1[i]) != 0:
            continue
        zoneName = names[i] if i < len(names) else ""
        if "interior" in zoneName.lower():
            continue
        start = int(minIds[i]) - 1
        end = int(maxIds[i])
        if end > start:
            selected.append(np.arange(start, end, dtype=np.int64))
    if not selected:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(selected)


def triangulateFaces(faceIdx: np.ndarray, offsets: np.ndarray, nnodes: np.ndarray, nodesFlat: np.ndarray) -> np.ndarray:
    tris: list[tuple[int, int, int]] = []
    for idx in faceIdx.tolist():
        k = int(nnodes[idx])
        if k < 3:
            continue
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        faceNodes = nodesFlat[start:end].astype(np.int64) - 1
        n0 = int(faceNodes[0])
        for j in range(1, k - 1):
            tris.append((n0, int(faceNodes[j]), int(faceNodes[j + 1])))
    if not tris:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)


def showInteractiveSurface(casePath: Path) -> None:
    with h5py.File(casePath, "r") as f:
        coords = np.asarray(f["meshes/1/nodes/coords/1"][()], dtype=np.float64)[:, :3]
        nnodes = np.asarray(f["meshes/1/faces/nodes/1/nnodes"][()], dtype=np.int64)
        nodesFlat = np.asarray(f["meshes/1/faces/nodes/1/nodes"][()], dtype=np.int64)
        z = f["meshes/1/faces/zoneTopology"]
        zoneTopo = {
            "name": np.asarray(z["name"][()]),
            "minId": np.asarray(z["minId"][()]),
            "maxId": np.asarray(z["maxId"][()]),
            "c1": np.asarray(z["c1"][()]),
        }

    offsets = faceOffsets(nnodes)
    faceIdx = collectBoundaryFaceIndices(zoneTopo)
    triangles = triangulateFaces(faceIdx, offsets, nnodes, nodesFlat)

    maxTriangles = 120000
    if triangles.shape[0] > maxTriangles:
        rng = np.random.default_rng(0)
        pick = rng.choice(triangles.shape[0], size=maxTriangles, replace=False)
        triangles = triangles[pick]

    triVerts = coords[triangles]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(
        triVerts,
        facecolor=(0.78, 0.78, 0.8, 0.85),
        edgecolor=(0.25, 0.25, 0.25, 0.15),
        linewidth=0.1,
    )
    ax.add_collection3d(poly)

    minVals = coords.min(axis=0)
    maxVals = coords.max(axis=0)
    center = (minVals + maxVals) * 0.5
    span = (maxVals - minVals).max() * 0.55
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)

    ax.set_title(f"interactive surface mesh | faces={faceIdx.size} tris={triangles.shape[0]}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    showInteractiveSurface(Path(r"F:/SIG/dataPreprocess/sampleData/1.cas.h5"))
