from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from preprocess.viz.meshPlot import subsampleCoordsAndScalar


def renderNodePointCloudMatplotlib(
    coords: np.ndarray,
    outputPath: Path,
    scalar: np.ndarray | None = None,
    maxPoints: int = 250000,
    markerSize: float = 0.15,
    elev: float = 20.0,
    azim: float = 45.0,
) -> None:
    pts, s = subsampleCoordsAndScalar(coords, scalar, maxPoints=maxPoints)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if s is None:
        c = pts[:, 2]
    else:
        c = s
    p = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=markerSize, cmap="viridis", linewidths=0)
    fig.colorbar(p, ax=ax, shrink=0.7, pad=0.08)
    ax.set_title("mesh node point cloud")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(str(outputPath), dpi=180)
    plt.close(fig)


def runMeshVizMatplotlib(config: dict[str, Any]) -> dict[str, Any]:
    from preprocess.viz.h5Scalars import readCoordsNx3

    casePath = Path(str(config["casePath"]).replace("\\", "/"))
    coordCandidates = list(config.get("nodeCoordPaths", ["meshes/1/nodes/coords/1"]))
    outImage = Path(str(config["outputImage"]).replace("\\", "/"))
    maxPoints = int(config.get("maxPoints", 250000))
    markerSize = float(config.get("markerSize", 0.15))
    elev = float(config.get("elev", 20.0))
    azim = float(config.get("azim", 45.0))
    coords = readCoordsNx3(casePath, coordCandidates)
    renderNodePointCloudMatplotlib(
        coords=coords,
        outputPath=outImage,
        scalar=None,
        maxPoints=maxPoints,
        markerSize=markerSize,
        elev=elev,
        azim=azim,
    )
    return {"kind": "meshPointCloudMatplotlib", "nNodes": int(coords.shape[0]), "outputImage": str(outImage)}
