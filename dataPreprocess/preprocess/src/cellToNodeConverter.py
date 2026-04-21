from pathlib import Path
from typing import Any

import h5py
import numpy as np

from preprocess.src.ioLayer import loadJson, resolveDataset, resolveDatasetOptional


def _normalizeValues(values: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(values, dtype=np.float64)
    isScalar = arr.ndim == 1
    if isScalar:
        arr = arr.reshape(-1, 1)
    return arr, isScalar


def _reduceByConnectivity(
    sourceValues: np.ndarray,
    targetSourceIds: list[np.ndarray],
    nTarget: int,
    method: str = "mean",
    sourceWeights: np.ndarray | None = None,
    targetSourceWeights: list[np.ndarray] | None = None,
) -> np.ndarray:
    if method not in {"mean", "weighted"}:
        raise ValueError(f"unsupported method: {method}")
    values, isScalar = _normalizeValues(sourceValues)
    nDim = values.shape[1]
    sumAtTarget = np.zeros((nTarget, nDim), dtype=np.float64)
    weightAtTarget = np.zeros(nTarget, dtype=np.float64)
    for targetId, sourceIdsRaw in enumerate(targetSourceIds):
        sourceIds = np.asarray(sourceIdsRaw).reshape(-1)
        if sourceIds.size == 0:
            continue
        if method == "mean":
            weights = np.ones(sourceIds.size, dtype=np.float64)
        else:
            if targetSourceWeights is not None and targetSourceWeights[targetId].size > 0:
                weights = np.asarray(targetSourceWeights[targetId], dtype=np.float64).reshape(-1)
            elif sourceWeights is not None:
                weights = np.asarray(sourceWeights, dtype=np.float64)[sourceIds]
            else:
                raise ValueError("weighted method requires sourceWeights or targetSourceWeights")
            if weights.size != sourceIds.size:
                raise ValueError("weights size mismatch with connectivity size")
        for localIdx, rawSourceId in enumerate(sourceIds):
            sourceId = int(rawSourceId)
            if sourceId < 0 or sourceId >= values.shape[0]:
                continue
            w = float(weights[localIdx])
            sumAtTarget[targetId] += values[sourceId] * w
            weightAtTarget[targetId] += w
    out = np.full((nTarget, nDim), np.nan, dtype=np.float64)
    valid = weightAtTarget > 0
    out[valid] = sumAtTarget[valid] / weightAtTarget[valid, np.newaxis]
    if isScalar:
        return out.reshape(-1)
    return out


def buildPointCellIds(cellNodeIds: list[np.ndarray], nNodes: int) -> list[np.ndarray]:
    pointCells: list[list[int]] = [[] for _ in range(nNodes)]
    for cellId, nodeIds in enumerate(cellNodeIds):
        for rawNodeId in np.asarray(nodeIds).reshape(-1):
            nodeId = int(rawNodeId)
            if 0 <= nodeId < nNodes:
                pointCells[nodeId].append(cellId)
    return [np.asarray(ids, dtype=np.int64) for ids in pointCells]


def pointFieldToCell(
    pointValues: np.ndarray,
    cellNodeIds: list[np.ndarray],
    method: str = "mean",
    pointWeights: np.ndarray | None = None,
    cellNodeWeights: list[np.ndarray] | None = None,
) -> np.ndarray:
    return _reduceByConnectivity(
        sourceValues=pointValues,
        targetSourceIds=cellNodeIds,
        nTarget=len(cellNodeIds),
        method=method,
        sourceWeights=pointWeights,
        targetSourceWeights=cellNodeWeights,
    )


def cellFieldToPoint(
    cellValues: np.ndarray,
    cellNodeIds: list[np.ndarray],
    nNodes: int,
    method: str = "mean",
    cellWeights: np.ndarray | None = None,
    pointCellWeights: list[np.ndarray] | None = None,
) -> np.ndarray:
    pointCellIds = buildPointCellIds(cellNodeIds, nNodes)
    return _reduceByConnectivity(
        sourceValues=cellValues,
        targetSourceIds=pointCellIds,
        nTarget=nNodes,
        method=method,
        sourceWeights=cellWeights,
        targetSourceWeights=pointCellWeights,
    )


def cellFieldToNodeMean(
    cellValues: np.ndarray,
    cellNodeIds: list[np.ndarray],
    nNodes: int,
) -> np.ndarray:
    return cellFieldToPoint(cellValues, cellNodeIds, nNodes, method="mean")


def computeCellCenters(pointCoords: np.ndarray, cellNodeIds: list[np.ndarray], method: str = "mean") -> np.ndarray:
    return pointFieldToCell(pointCoords, cellNodeIds, method=method)


def cellFieldToNodeWeighted(
    cellValues: np.ndarray,
    cellNodeIds: list[np.ndarray],
    nNodes: int,
    cellWeights: np.ndarray | None = None,
    pointCellWeights: list[np.ndarray] | None = None,
) -> np.ndarray:
    return cellFieldToPoint(
        cellValues=cellValues,
        cellNodeIds=cellNodeIds,
        nNodes=nNodes,
        method="weighted",
        cellWeights=cellWeights,
        pointCellWeights=pointCellWeights,
    )


def pointFieldToCellMean(
    pointValues: np.ndarray,
    cellNodeIds: list[np.ndarray],
) -> np.ndarray:
    return pointFieldToCell(pointValues, cellNodeIds, method="mean")


def pointFieldToCellWeighted(
    pointValues: np.ndarray,
    cellNodeIds: list[np.ndarray],
    pointWeights: np.ndarray | None = None,
    cellNodeWeights: list[np.ndarray] | None = None,
) -> np.ndarray:
    return pointFieldToCell(
        pointValues=pointValues,
        cellNodeIds=cellNodeIds,
        method="weighted",
        pointWeights=pointWeights,
        cellNodeWeights=cellNodeWeights,
    )


def loadCellNodeIdsFromCsr(
    casePath: Path,
    offsetCandidates: list[str],
    indexCandidates: list[str],
) -> tuple[list[np.ndarray], int]:
    with h5py.File(casePath, "r") as meshFile:
        offsets = resolveDataset(meshFile, offsetCandidates).reshape(-1)
        indices = resolveDataset(meshFile, indexCandidates).reshape(-1)
    nCells = int(offsets.size - 1)
    if nCells < 0:
        raise ValueError("invalid CSR offsets length")
    cellNodeIds: list[np.ndarray] = []
    maxNodeId = -1
    for cellIdx in range(nCells):
        start = int(offsets[cellIdx])
        end = int(offsets[cellIdx + 1])
        block = indices[start:end].astype(np.int64, copy=False)
        cellNodeIds.append(block)
        if block.size > 0:
            maxNodeId = max(maxNodeId, int(block.max()))
    nNodes = maxNodeId + 1 if maxNodeId >= 0 else 0
    return cellNodeIds, nNodes


def loadNodeCountFromCoords(casePath: Path, coordCandidates: list[str]) -> int:
    with h5py.File(casePath, "r") as meshFile:
        coords = resolveDataset(meshFile, coordCandidates)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must be 2D with shape (nNodes, dim)")
    return int(coords.shape[0])


def loadCellNodeIdsFromMeshConfig(casePath: Path, meshConfig: dict[str, Any]) -> tuple[list[np.ndarray], int]:
    mode = str(meshConfig.get("mode", "csr"))
    casePathObj = Path(casePath)
    if mode == "csr":
        offsetCandidates = meshConfig.get("cellNodeOffsetPaths", [])
        indexCandidates = meshConfig.get("cellNodeIndexPaths", [])
        if not offsetCandidates or not indexCandidates:
            raise ValueError("csr mode requires cellNodeOffsetPaths and cellNodeIndexPaths")
        cellNodeIds, nNodesFromConn = loadCellNodeIdsFromCsr(
            casePathObj,
            list(offsetCandidates),
            list(indexCandidates),
        )
        coordCandidates = meshConfig.get("nodeCoordPaths", [])
        if "nNodes" in meshConfig and meshConfig["nNodes"] is not None:
            nNodes = int(meshConfig["nNodes"])
        elif coordCandidates:
            nNodesCoords = loadNodeCountFromCoords(casePathObj, list(coordCandidates))
            nNodes = max(nNodesFromConn, nNodesCoords)
        else:
            nNodes = nNodesFromConn
        return cellNodeIds, nNodes
    raise ValueError(f"unsupported mesh mode: {mode}")


class cellToNodeConverter:
    def __init__(self, meshConfigPath: str | Path):
        self.meshConfigPath = Path(meshConfigPath)
        self.meshConfig = loadJson(self.meshConfigPath)

    def convertFromCase(
        self,
        casePath: str | Path,
        cellValues: np.ndarray,
        method: str = "mean",
        cellWeights: np.ndarray | None = None,
        pointCellWeights: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        casePathObj = Path(casePath)
        cellNodeIds, nNodes = loadCellNodeIdsFromMeshConfig(casePathObj, self.meshConfig)
        return cellFieldToPoint(
            cellValues=cellValues,
            cellNodeIds=cellNodeIds,
            nNodes=nNodes,
            method=method,
            cellWeights=cellWeights,
            pointCellWeights=pointCellWeights,
        )

    @staticmethod
    def probeOffsetsAndIndices(casePath: str | Path, meshConfig: dict[str, Any]) -> dict[str, Any]:
        casePathObj = Path(casePath)
        offsetCandidates = list(meshConfig.get("cellNodeOffsetPaths", []))
        indexCandidates = list(meshConfig.get("cellNodeIndexPaths", []))
        report: dict[str, Any] = {"casePath": str(casePathObj), "offsetsFound": None, "indicesFound": None}
        with h5py.File(casePathObj, "r") as meshFile:
            if offsetCandidates:
                found = resolveDatasetOptional(meshFile, offsetCandidates)
                if found is not None:
                    report["offsetsFound"] = list(found.shape)
            if indexCandidates:
                found = resolveDatasetOptional(meshFile, indexCandidates)
                if found is not None:
                    report["indicesFound"] = list(found.shape)
        return report


class meshFieldConverter:
    def __init__(self, meshConfigPath: str | Path):
        self.meshConfigPath = Path(meshConfigPath)
        self.meshConfig = loadJson(self.meshConfigPath)
    def _loadConnectivity(self, casePath: str | Path) -> tuple[list[np.ndarray], int]:
        return loadCellNodeIdsFromMeshConfig(Path(casePath), self.meshConfig)
    def pointToCell(
        self,
        casePath: str | Path,
        pointValues: np.ndarray,
        method: str = "mean",
        pointWeights: np.ndarray | None = None,
        cellNodeWeights: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        cellNodeIds, _ = self._loadConnectivity(casePath)
        return pointFieldToCell(
            pointValues=pointValues,
            cellNodeIds=cellNodeIds,
            method=method,
            pointWeights=pointWeights,
            cellNodeWeights=cellNodeWeights,
        )
    def cellToPoint(
        self,
        casePath: str | Path,
        cellValues: np.ndarray,
        method: str = "mean",
        cellWeights: np.ndarray | None = None,
        pointCellWeights: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        cellNodeIds, nNodes = self._loadConnectivity(casePath)
        return cellFieldToPoint(
            cellValues=cellValues,
            cellNodeIds=cellNodeIds,
            nNodes=nNodes,
            method=method,
            cellWeights=cellWeights,
            pointCellWeights=pointCellWeights,
        )
    def cellCenters(self, casePath: str | Path, pointCoords: np.ndarray, method: str = "mean") -> np.ndarray:
        cellNodeIds, _ = self._loadConnectivity(casePath)
        return computeCellCenters(pointCoords, cellNodeIds, method=method)
