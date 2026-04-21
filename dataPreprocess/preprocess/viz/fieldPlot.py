from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def renderScalarHistogram(values: np.ndarray, outputPath: Path, title: str = "scalar") -> None:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(v, bins=80, color="#4c72b0", edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(str(outputPath), dpi=150)
    plt.close(fig)


def renderCellPointCloud(
    cellCentroids: np.ndarray,
    cellScalars: np.ndarray,
    outputPath: Path,
    scalarName: str = "field",
    maxPoints: int = 200000,
    pointSize: int = 3,
) -> None:
    from preprocess.viz.meshPlot import renderNodePointCloud

    renderNodePointCloud(
        cellCentroids,
        outputPath,
        scalar=cellScalars,
        scalarName=scalarName,
        maxPoints=maxPoints,
        pointSize=pointSize,
    )


def runFieldViz(config: dict[str, Any]) -> dict[str, Any]:
    from preprocess.viz.h5Scalars import readDataset1d, readNpyPoints

    datPath = Path(str(config["datPath"]).replace("\\", "/"))
    datasetPath = str(config["scalarDatasetPath"])
    title = config.get("histTitle", datasetPath)
    histOut = Path(str(config["histOutput"]).replace("\\", "/"))
    values = readDataset1d(datPath, datasetPath)
    renderScalarHistogram(values, histOut, title=title)
    report: dict[str, Any] = {
        "kind": "field",
        "nCells": int(values.size),
        "histOutput": str(histOut),
    }
    cloudOut = config.get("cloudOutput")
    centroidsPath = config.get("cellCentroidsNpy")
    if cloudOut and centroidsPath:
        cents = readNpyPoints(Path(str(centroidsPath).replace("\\", "/")))
        if cents.shape[0] != values.size:
            raise ValueError("cellCentroidsNpy rows must match scalar length")
        cloudPath = Path(str(cloudOut).replace("\\", "/"))
        renderCellPointCloud(
            cents,
            values,
            cloudPath,
            scalarName=config.get("scalarName", "field"),
            maxPoints=int(config.get("maxPoints", 200000)),
        )
        report["cloudOutput"] = str(cloudPath)
    return report
