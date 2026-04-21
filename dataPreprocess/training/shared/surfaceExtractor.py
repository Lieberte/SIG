import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def loadTopologyMeta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def loadSchema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def filterZones(
    topologyMeta: dict[str, Any],
    role: str | None = None,
    zoneType: str | None = None,
) -> list[dict[str, Any]]:
    zones = topologyMeta.get("faceZones", [])
    filtered: list[dict[str, Any]] = []
    for z in zones:
        cat = z.get("category", {})
        if role is not None and cat.get("role") != role:
            continue
        if zoneType is not None and cat.get("type") != zoneType:
            continue
        filtered.append(z)
    return filtered


def loadFaceOwnerCells(casePath: Path) -> np.ndarray:
    with h5py.File(casePath, "r") as f:
        base = "meshes/1/faces/c0"
        if f"{base}/1" in f:
            return np.asarray(f[f"{base}/1"][()]).reshape(-1).astype(np.int64)
        if base in f:
            node = f[base]
            if isinstance(node, h5py.Dataset):
                return np.asarray(node[()]).reshape(-1).astype(np.int64)
            partKeys = sorted(
                node.keys(),
                key=lambda x: int(x) if x.isdigit() else 9999,
            )
            parts = [np.asarray(node[k][()]).reshape(-1) for k in partKeys]
            return np.concatenate(parts).astype(np.int64)
    raise ValueError("faces/c0 not found in cas.h5")


def cellIdsForZones(
    faceOwnerCells: np.ndarray,
    zones: list[dict[str, Any]],
    maxCellId: int | None = None,
) -> np.ndarray:
    cellIdSet: set[int] = set()
    for z in zones:
        start = int(z["minFaceId"]) - 1
        end = int(z["maxFaceId"])
        if start < 0:
            start = 0
        if end > faceOwnerCells.size:
            end = faceOwnerCells.size
        ownerCells = faceOwnerCells[start:end] - 1
        validMask = ownerCells >= 0
        if maxCellId is not None:
            validMask &= ownerCells < maxCellId
        cellIdSet.update(ownerCells[validMask].tolist())
    if not cellIdSet:
        return np.array([], dtype=np.int64)
    return np.array(sorted(cellIdSet), dtype=np.int64)


def sliceFieldFromMatrix(
    matrix: np.ndarray,
    schema: dict[str, Any],
    fieldName: str,
    cellIds: np.ndarray,
) -> np.ndarray:
    fieldOrder = schema["fieldOrder"]
    nCells = schema["nCells"]
    if fieldName not in fieldOrder:
        raise KeyError(f"field '{fieldName}' not in fieldOrder: {fieldOrder}")
    fieldIdx = fieldOrder.index(fieldName)
    startRow = fieldIdx * nCells
    endRow = startRow + nCells
    fieldBlock = matrix[startRow:endRow, :]
    validMask = (cellIds >= 0) & (cellIds < nCells)
    validCellIds = cellIds[validMask]
    return fieldBlock[validCellIds, :]


def zoneMeanFromMatrix(
    matrix: np.ndarray,
    schema: dict[str, Any],
    cellIds: np.ndarray,
    fieldNames: list[str],
) -> np.ndarray:
    nSnapshots = matrix.shape[1]
    result = np.zeros((nSnapshots, len(fieldNames)), dtype=np.float64)
    for i, name in enumerate(fieldNames):
        values = sliceFieldFromMatrix(matrix, schema, name, cellIds)
        result[:, i] = np.nanmean(values, axis=0)
    return result


def buildTrainingPair(
    matrix: np.ndarray,
    schema: dict[str, Any],
    topologyMeta: dict[str, Any],
    faceOwnerCells: np.ndarray,
    bcFieldNames: list[str],
    targetFieldName: str,
    inletRole: str = "boundary",
    inletType: str = "inlet",
    wallRole: str = "boundary",
    wallType: str = "wall",
) -> dict[str, Any]:
    nCells = schema["nCells"]
    inletZones = filterZones(topologyMeta, role=inletRole, zoneType=inletType)
    wallZones = filterZones(topologyMeta, role=wallRole, zoneType=wallType)
    inletCellIds = cellIdsForZones(faceOwnerCells, inletZones, maxCellId=nCells)
    wallCellIds = cellIdsForZones(faceOwnerCells, wallZones, maxCellId=nCells)
    bcInput = zoneMeanFromMatrix(matrix, schema, inletCellIds, bcFieldNames)
    surfaceTarget = sliceFieldFromMatrix(matrix, schema, targetFieldName, wallCellIds)
    return {
        "bcInput": bcInput,
        "surfaceTarget": surfaceTarget.T,
        "inletCellIds": inletCellIds,
        "wallCellIds": wallCellIds,
    }
