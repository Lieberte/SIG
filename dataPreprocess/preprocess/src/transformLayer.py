from typing import Any

import numpy as np


def buildCanonicalIndex(caseMeta: dict[str, np.ndarray], datFrame: dict[str, Any]) -> np.ndarray:
    casCellIds = caseMeta.get("casCellIds")
    datCellIds = datFrame.get("indexes", {}).get("datCellIds")
    if casCellIds is not None and casCellIds.size > 0:
        return casCellIds.astype(np.int64)
    if datCellIds is not None and datCellIds.size > 0:
        return datCellIds.astype(np.int64)
    key = datFrame.get("canonicalFromField")
    if key and key in datFrame["fields"]:
        firstField = datFrame["fields"][key]
    else:
        firstFieldName = next(iter(datFrame["fields"]))
        firstField = datFrame["fields"][firstFieldName]
    firstArr = np.asarray(firstField)
    nCells = int(firstArr.shape[0])
    return np.arange(nCells, dtype=np.int64)


def buildDatToCanonicalMap(canonicalCellIds: np.ndarray, datCellIds: np.ndarray | None) -> np.ndarray:
    if datCellIds is None or datCellIds.size == 0:
        return np.arange(canonicalCellIds.size, dtype=np.int64)
    mapper = np.full(canonicalCellIds.shape[0], -1, dtype=np.int64)
    datIndex = {int(cellId): idx for idx, cellId in enumerate(datCellIds.tolist())}
    for idx, cellId in enumerate(canonicalCellIds.tolist()):
        mapper[idx] = datIndex.get(int(cellId), -1)
    return mapper


def fillValueToken(token: str) -> float:
    if token.lower() == "nan":
        return float("nan")
    return float(token)


def alignField(rawField: np.ndarray, datToCanonicalMap: np.ndarray, fillValue: float) -> np.ndarray:
    arr = np.asarray(rawField)
    if arr.ndim == 2 and arr.shape[1] > 1:
        arr = arr[:, 0]
    flat = arr.reshape(-1)
    target = np.full(datToCanonicalMap.shape[0], fillValue, dtype=np.float64)
    validMask = datToCanonicalMap >= 0
    target[validMask] = flat[datToCanonicalMap[validMask]]
    return target


def alignFrame(caseMeta: dict[str, np.ndarray], datFrame: dict[str, Any], fillValue: float) -> dict[str, Any]:
    canonicalCellIds = buildCanonicalIndex(caseMeta, datFrame)
    datCellIds = datFrame.get("indexes", {}).get("datCellIds")
    datToCanonicalMap = buildDatToCanonicalMap(canonicalCellIds, datCellIds)
    alignedFields: dict[str, np.ndarray] = {}
    for fieldName, rawArr in datFrame["fields"].items():
        alignedFields[fieldName] = alignField(np.asarray(rawArr).reshape(-1), datToCanonicalMap, fillValue)
    return {
        "canonicalCellIds": canonicalCellIds,
        "timeValue": datFrame.get("timeValue"),
        "dataPath": datFrame.get("dataPath"),
        "fields": alignedFields,
    }
