from typing import Any

import numpy as np
import pandas as pd


def buildSnapshot(alignedFrame: dict[str, Any], fieldOrder: list[str]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for fieldName in fieldOrder:
        if fieldName not in alignedFrame["fields"]:
            raise KeyError(f"field missing in aligned frame: {fieldName}")
        parts.append(np.asarray(alignedFrame["fields"][fieldName]).reshape(-1))
    return np.concatenate(parts, axis=0)


def assembleMatrix(alignedFrames: list[dict[str, Any]], fieldOrder: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    snapshotCols: list[np.ndarray] = []
    metadataRows: list[dict[str, Any]] = []
    for colIdx, frame in enumerate(alignedFrames):
        snapshot = buildSnapshot(frame, fieldOrder)
        snapshotCols.append(snapshot)
        metadataRows.append(
            {
                "colIdx": colIdx,
                "timeValue": frame.get("timeValue"),
                "dataPath": frame.get("dataPath"),
                "nCells": int(frame["canonicalCellIds"].size),
                "fieldOrder": ",".join(fieldOrder),
            }
        )
    matrix = np.column_stack(snapshotCols) if snapshotCols else np.zeros((0, 0), dtype=np.float64)
    metadata = pd.DataFrame(metadataRows)
    return matrix, metadata


def buildSchema(alignedFrames: list[dict[str, Any]], fieldOrder: list[str], matrix: np.ndarray) -> dict[str, Any]:
    nCells = int(alignedFrames[0]["canonicalCellIds"].size) if alignedFrames else 0
    return {
        "nCells": nCells,
        "nFields": len(fieldOrder),
        "fieldOrder": fieldOrder,
        "nSnapshots": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "matrixShape": list(matrix.shape),
    }
