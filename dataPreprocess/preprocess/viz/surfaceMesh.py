from pathlib import Path
from typing import Any

import h5py
import numpy as np


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


def collectBoundaryFaceIndices(zoneTopo: dict[str, np.ndarray], includeZoneNames: list[str] | None = None) -> np.ndarray:
    ids = np.asarray(zoneTopo["id"]).reshape(-1)
    minIds = np.asarray(zoneTopo["minId"]).reshape(-1)
    maxIds = np.asarray(zoneTopo["maxId"]).reshape(-1)
    c1 = np.asarray(zoneTopo["c1"]).reshape(-1)
    names = parseZoneNames(np.asarray(zoneTopo["name"]))
    selected: list[np.ndarray] = []
    for i in range(ids.size):
        if int(c1[i]) != 0:
            continue
        zoneName = names[i] if i < len(names) else f"zone_{int(ids[i])}"
        if includeZoneNames and zoneName not in includeZoneNames:
            continue
        start = int(minIds[i]) - 1
        end = int(maxIds[i])
        if end <= start:
            continue
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


def triangulateFacesWithFaceScalar(
    faceIdx: np.ndarray,
    offsets: np.ndarray,
    nnodes: np.ndarray,
    nodesFlat: np.ndarray,
    faceScalar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tris: list[tuple[int, int, int]] = []
    triScalar: list[float] = []
    for localFacePos, idx in enumerate(faceIdx.tolist()):
        k = int(nnodes[idx])
        if k < 3:
            continue
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        faceNodes = nodesFlat[start:end].astype(np.int64) - 1
        n0 = int(faceNodes[0])
        val = float(faceScalar[localFacePos])
        for j in range(1, k - 1):
            tris.append((n0, int(faceNodes[j]), int(faceNodes[j + 1])))
            triScalar.append(val)
    if not tris:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    return np.asarray(tris, dtype=np.int64), np.asarray(triScalar, dtype=np.float64)


def buildSurfaceFromCase(casePath: Path, includeZoneNames: list[str] | None = None) -> dict[str, Any]:
    with h5py.File(casePath, "r") as f:
        coords = np.asarray(f["meshes/1/nodes/coords/1"][()], dtype=np.float64)
        nnodes = np.asarray(f["meshes/1/faces/nodes/1/nnodes"][()], dtype=np.int64)
        nodesFlat = np.asarray(f["meshes/1/faces/nodes/1/nodes"][()], dtype=np.int64)
        z = f["meshes/1/faces/zoneTopology"]
        zoneTopo = {
            "id": np.asarray(z["id"][()]),
            "name": np.asarray(z["name"][()]),
            "minId": np.asarray(z["minId"][()]),
            "maxId": np.asarray(z["maxId"][()]),
            "c1": np.asarray(z["c1"][()]),
        }
    offsets = faceOffsets(nnodes)
    boundaryFaceIdx = collectBoundaryFaceIndices(zoneTopo, includeZoneNames=includeZoneNames)
    triangles = triangulateFaces(boundaryFaceIdx, offsets, nnodes, nodesFlat)
    return {
        "coords": coords[:, :3],
        "triangles": triangles,
        "boundaryFaceCount": int(boundaryFaceIdx.size),
    }


def renderSurfaceMesh(
    coords: np.ndarray,
    triangles: np.ndarray,
    outputPath: Path,
    wireframe: bool = True,
    color: str = "lightgray",
    background: str = "white",
) -> None:
    import pyvista as pv

    nTri = int(triangles.shape[0])
    faces = np.hstack([np.full((nTri, 1), 3, dtype=np.int64), triangles]).reshape(-1)
    mesh = pv.PolyData(coords, faces)
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1200))
    if wireframe:
        plotter.add_mesh(mesh, style="wireframe", color=color, line_width=1)
    else:
        plotter.add_mesh(mesh, color=color, show_edges=True)
    plotter.set_background(background)
    plotter.camera_position = "iso"
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(outputPath))
    plotter.close()


def runSurfaceViz(config: dict[str, Any]) -> dict[str, Any]:
    casePath = Path(str(config["casePath"]).replace("\\", "/"))
    outputImage = Path(str(config["outputImage"]).replace("\\", "/"))
    includeZoneNames = config.get("includeZoneNames")
    data = buildSurfaceFromCase(casePath, includeZoneNames=includeZoneNames)
    renderSurfaceMesh(
        coords=data["coords"],
        triangles=data["triangles"],
        outputPath=outputImage,
        wireframe=bool(config.get("wireframe", True)),
        color=str(config.get("color", "lightgray")),
        background=str(config.get("background", "white")),
    )
    triNpy = config.get("trianglesOutput")
    if triNpy:
        triPath = Path(str(triNpy).replace("\\", "/"))
        triPath.parent.mkdir(parents=True, exist_ok=True)
        np.save(triPath, data["triangles"])
    return {
        "kind": "surfaceMesh",
        "boundaryFaceCount": int(data["boundaryFaceCount"]),
        "triangleCount": int(data["triangles"].shape[0]),
        "outputImage": str(outputImage),
        "trianglesOutput": str(triNpy) if triNpy else None,
    }


def runSurfacePressureViz(config: dict[str, Any]) -> dict[str, Any]:
    import pyvista as pv

    casePath = Path(str(config["casePath"]).replace("\\", "/"))
    datPath = Path(str(config["datPath"]).replace("\\", "/"))
    pressurePath = str(config.get("pressureDatasetPath", "results/1/phase-1/cells/SV_P/1"))
    outputImage = Path(str(config["outputImage"]).replace("\\", "/"))
    includeZoneNames = config.get("includeZoneNames")

    with h5py.File(casePath, "r") as f:
        coords = np.asarray(f["meshes/1/nodes/coords/1"][()], dtype=np.float64)[:, :3]
        nnodes = np.asarray(f["meshes/1/faces/nodes/1/nnodes"][()], dtype=np.int64)
        nodesFlat = np.asarray(f["meshes/1/faces/nodes/1/nodes"][()], dtype=np.int64)
        c0 = np.asarray(f["meshes/1/faces/c0/1"][()], dtype=np.int64)
        z = f["meshes/1/faces/zoneTopology"]
        zoneTopo = {
            "id": np.asarray(z["id"][()]),
            "name": np.asarray(z["name"][()]),
            "minId": np.asarray(z["minId"][()]),
            "maxId": np.asarray(z["maxId"][()]),
            "c1": np.asarray(z["c1"][()]),
        }
    with h5py.File(datPath, "r") as f:
        pressure = np.asarray(f[pressurePath][()], dtype=np.float64).reshape(-1)

    offsets = faceOffsets(nnodes)
    faceIdx = collectBoundaryFaceIndices(zoneTopo, includeZoneNames=includeZoneNames)
    owner = c0[faceIdx] - 1
    faceP = np.full(faceIdx.size, np.nan, dtype=np.float64)
    valid = (owner >= 0) & (owner < pressure.size)
    faceP[valid] = pressure[owner[valid]]
    triangles, triPressure = triangulateFacesWithFaceScalar(faceIdx, offsets, nnodes, nodesFlat, faceP)

    nTri = int(triangles.shape[0])
    faces = np.hstack([np.full((nTri, 1), 3, dtype=np.int64), triangles]).reshape(-1)
    mesh = pv.PolyData(coords, faces)
    mesh.cell_data["pressure"] = triPressure

    outputImage.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1200))
    plotter.add_mesh(mesh, scalars="pressure", cmap="turbo", show_edges=False, nan_color="lightgray")
    plotter.set_background("white")
    plotter.camera_position = "iso"
    plotter.screenshot(str(outputImage))
    plotter.close()

    return {
        "kind": "surfacePressure",
        "pressureSize": int(pressure.size),
        "boundaryFaceCount": int(faceIdx.size),
        "triangleCount": int(triangles.shape[0]),
        "validFacePressureCount": int(np.isfinite(faceP).sum()),
        "outputImage": str(outputImage),
    }
