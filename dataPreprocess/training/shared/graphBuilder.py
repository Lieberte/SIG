import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

EDGE_TYPE_INTERNAL = 0
EDGE_TYPE_INTERFACE = 1
EDGE_TYPE_BOUNDARY = 2


def _loadPartitionedArray(h5File: h5py.File, basePath: str) -> np.ndarray | None:
    if f"{basePath}/1" in h5File:
        node = h5File[basePath]
        partKeys = sorted(
            node.keys(),
            key=lambda x: int(x) if x.isdigit() else 9999,
        )
        parts = [np.asarray(node[k][()]).reshape(-1) for k in partKeys]
        return np.concatenate(parts).astype(np.int64)
    if basePath in h5File:
        node = h5File[basePath]
        if isinstance(node, h5py.Dataset):
            return np.asarray(node[()]).reshape(-1).astype(np.int64)
    return None


def _classifyFaceEdge(faceIdx: int, zoneRanges: list[dict]) -> int:
    for zr in zoneRanges:
        if zr["start"] <= faceIdx < zr["end"]:
            return zr["edgeType"]
    return EDGE_TYPE_INTERNAL


def _buildZoneRanges(topologyMeta: dict[str, Any]) -> list[dict]:
    ranges: list[dict] = []
    for z in topologyMeta.get("faceZones", []):
        cat = z.get("category", {})
        role = cat.get("role", "")
        start = int(z["minFaceId"]) - 1
        end = int(z["maxFaceId"])
        if role == "interface":
            edgeType = EDGE_TYPE_INTERFACE
        elif role == "boundary":
            edgeType = EDGE_TYPE_BOUNDARY
        else:
            edgeType = EDGE_TYPE_INTERNAL
        ranges.append({"start": start, "end": end, "edgeType": edgeType})
    return ranges


def buildCellAdjacencyFast(casePath: Path) -> dict[str, np.ndarray]:
    with h5py.File(casePath, "r") as f:
        c0 = _loadPartitionedArray(f, "meshes/1/faces/c0")
        c1 = _loadPartitionedArray(f, "meshes/1/faces/c1")
    if c0 is None:
        raise ValueError("faces/c0 not found in cas.h5")
    if c1 is None:
        return {"edgeIndex": np.zeros((2, 0), dtype=np.int64), "nCells": 0, "nEdges": 0}
    nPair = min(c0.size, c1.size)
    a = c0[:nPair] - 1
    b = c1[:nPair] - 1
    mask = (a >= 0) & (b >= 0)
    a = a[mask]
    b = b[mask]
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    edgeIndex = np.stack([src, dst])
    nCells = int(max(src.max(), dst.max())) + 1 if src.size > 0 else 0
    return {"edgeIndex": edgeIndex, "nCells": nCells, "nEdges": int(src.size)}


def buildCellAdjacencyTyped(
    casePath: Path,
    topologyMeta: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    with h5py.File(casePath, "r") as f:
        c0 = _loadPartitionedArray(f, "meshes/1/faces/c0")
        c1 = _loadPartitionedArray(f, "meshes/1/faces/c1")
    if c0 is None:
        raise ValueError("faces/c0 not found in cas.h5")
    if c1 is None:
        return {
            "edgeIndex": np.zeros((2, 0), dtype=np.int64),
            "edgeType": np.zeros((0,), dtype=np.int64),
            "nCells": 0,
            "nEdges": 0,
        }
    nPair = min(c0.size, c1.size)
    a = c0[:nPair] - 1
    b = c1[:nPair] - 1
    mask = (a >= 0) & (b >= 0)
    faceIndices = np.where(mask)[0]
    a = a[mask]
    b = b[mask]
    zoneRanges = _buildZoneRanges(topologyMeta) if topologyMeta else []
    if zoneRanges:
        faceTypes = np.array(
            [_classifyFaceEdge(int(fi), zoneRanges) for fi in faceIndices],
            dtype=np.int64,
        )
    else:
        faceTypes = np.zeros(a.size, dtype=np.int64)
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    edgeType = np.concatenate([faceTypes, faceTypes])
    edgeIndex = np.stack([src, dst])
    nCells = int(max(src.max(), dst.max())) + 1 if src.size > 0 else 0
    return {
        "edgeIndex": edgeIndex,
        "edgeType": edgeType,
        "nCells": nCells,
        "nEdges": int(src.size),
    }


def buildEdgeAttr(edgeType: np.ndarray, nTypes: int = 3) -> np.ndarray:
    oneHot = np.zeros((edgeType.size, nTypes), dtype=np.float32)
    for i, t in enumerate(edgeType):
        if 0 <= t < nTypes:
            oneHot[i, t] = 1.0
    return oneHot


def extractNodeFeatures(
    matrix: np.ndarray,
    schema: dict[str, Any],
    fieldNames: list[str] | None = None,
) -> np.ndarray:
    fieldOrder = schema["fieldOrder"]
    nCells = schema["nCells"]
    if fieldNames is None:
        fieldNames = fieldOrder
    nSnapshots = matrix.shape[1]
    nFields = len(fieldNames)
    features = np.zeros((nSnapshots, nCells, nFields), dtype=np.float64)
    for i, name in enumerate(fieldNames):
        idx = fieldOrder.index(name)
        start = idx * nCells
        end = start + nCells
        features[:, :, i] = matrix[start:end, :].T
    return features


def buildPygData(
    casePath: Path,
    nodeFeatures: np.ndarray | None = None,
    topologyMeta: dict[str, Any] | None = None,
) -> "torch_geometric.data.Data":
    import torch
    from torch_geometric.data import Data
    graph = buildCellAdjacencyTyped(casePath, topologyMeta)
    edgeIndex = torch.from_numpy(graph["edgeIndex"])
    edgeType = graph["edgeType"]
    edgeAttrNp = buildEdgeAttr(edgeType)
    edgeAttr = torch.from_numpy(edgeAttrNp)
    x = torch.from_numpy(nodeFeatures.astype(np.float32)) if nodeFeatures is not None else None
    return Data(x=x, edge_index=edgeIndex, edge_attr=edgeAttr, num_nodes=graph["nCells"])


def saveCellGraph(casePath: Path, outPath: Path, topologyMeta: dict[str, Any] | None = None) -> dict[str, Any]:
    graph = buildCellAdjacencyTyped(casePath, topologyMeta)
    outPath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        outPath,
        edgeIndex=graph["edgeIndex"],
        edgeType=graph["edgeType"],
    )
    return {"nCells": graph["nCells"], "nEdges": graph["nEdges"], "outPath": str(outPath)}
