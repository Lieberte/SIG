from pathlib import Path
from typing import Any

import numpy as np


def subsampleCoordsAndScalar(
    coords: np.ndarray,
    scalar: np.ndarray | None,
    maxPoints: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    n = coords.shape[0]
    if n <= maxPoints:
        return coords, scalar
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=maxPoints, replace=False)
    c = coords[idx]
    s = None
    if scalar is not None:
        s = np.asarray(scalar, dtype=np.float64).reshape(-1)[idx]
    return c, s


def renderNodePointCloud(
    coords: np.ndarray,
    outputPath: Path,
    scalar: np.ndarray | None = None,
    scalarName: str = "scalar",
    maxPoints: int = 300000,
    pointSize: int = 2,
    windowSize: tuple[int, int] = (1200, 900),
) -> None:
    import pyvista as pv

    pts, s = subsampleCoordsAndScalar(coords, scalar, maxPoints)
    cloud = pv.PolyData(pts)
    if s is not None:
        cloud[scalarName] = s
        colorName = scalarName
    else:
        cloud["height"] = pts[:, 2]
        colorName = "height"
    plotter = pv.Plotter(off_screen=True, window_size=windowSize)
    plotter.add_mesh(
        cloud,
        scalars=colorName,
        point_size=pointSize,
        render_points_as_spheres=False,
        cmap="viridis",
    )
    plotter.camera_position = "iso"
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(outputPath))
    plotter.close()


def runMeshViz(config: dict[str, Any]) -> dict[str, Any]:
    from preprocess.viz.h5Scalars import readCoordsNx3

    casePath = Path(str(config["casePath"]).replace("\\", "/"))
    coordCandidates = list(config.get("nodeCoordPaths", ["meshes/1/nodes/coords/1"]))
    outImage = Path(str(config["outputImage"]).replace("\\", "/"))
    maxPoints = int(config.get("maxPoints", 300000))
    coords = readCoordsNx3(casePath, coordCandidates)
    renderNodePointCloud(coords, outImage, scalar=None, maxPoints=maxPoints)
    return {"kind": "meshPointCloud", "nNodes": int(coords.shape[0]), "outputImage": str(outImage)}
