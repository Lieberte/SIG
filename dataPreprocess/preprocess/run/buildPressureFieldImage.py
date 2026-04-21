import json
import sys
from pathlib import Path

import h5py
import numpy as np


def addProjectRootToPath() -> None:
    currentFile = Path(__file__).resolve()
    projectRoot = currentFile.parents[2]
    if str(projectRoot) not in sys.path:
        sys.path.insert(0, str(projectRoot))


addProjectRootToPath()

from preprocess.viz.fieldPlot import renderCellPointCloud
from preprocess.viz.h5Scalars import readDataset1d


def reconstructCellCentroids(casePath: Path, nCells: int) -> np.ndarray:
    with h5py.File(casePath, "r") as f:
        c0 = f["meshes/1/faces/c0/1"][()]
        nnodes = f["meshes/1/faces/nodes/1/nnodes"][()]
        nodes = f["meshes/1/faces/nodes/1/nodes"][()]
        coords = f["meshes/1/nodes/coords/1"][()]
    sums = np.zeros((nCells, 3), dtype=np.float64)
    counts = np.zeros(nCells, dtype=np.int64)
    pos = 0
    nFaces = int(c0.shape[0])
    for i in range(nFaces):
        k = int(nnodes[i])
        faceNodeIds = nodes[pos:pos + k].astype(np.int64) - 1
        pos += k
        faceCentroid = coords[faceNodeIds, :3].mean(axis=0)
        a = int(c0[i]) - 1
        if 0 <= a < nCells:
            sums[a] += faceCentroid
            counts[a] += 1
    out = np.full((nCells, 3), np.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid, None]
    return out


def main() -> None:
    root = Path(r"F:/SIG/dataPreprocess")
    casePath = root / "sampleData/1.cas.h5"
    datPath = root / "sampleData/1.dat.h5"
    outDir = root / "preprocess/out"
    outDir.mkdir(parents=True, exist_ok=True)
    pressure = readDataset1d(datPath, "results/1/phase-1/cells/SV_P/1")
    nCells = int(pressure.size)
    centroids = reconstructCellCentroids(casePath, nCells)
    np.save(outDir / "cellCentroids_fromFaces.npy", centroids)
    renderCellPointCloud(
        cellCentroids=centroids,
        cellScalars=pressure,
        outputPath=outDir / "viz_pressure_field_cloud.png",
        scalarName="pressure",
        maxPoints=200000,
        pointSize=2,
    )
    report = {
        "nCells": nCells,
        "centroidsFile": str(outDir / "cellCentroids_fromFaces.npy"),
        "pressureFieldImage": str(outDir / "viz_pressure_field_cloud.png"),
        "nanCentroidCount": int(np.isnan(centroids[:, 0]).sum()),
    }
    (outDir / "pressureFieldReport.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
