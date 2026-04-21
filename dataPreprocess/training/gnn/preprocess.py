"""
GNN specific data preprocessing.

- Extracts node features from snapshot matrix
- Builds adjacency graph (typed)
- Supports temporal splits (no leakage)
"""

from __future__ import annotations

import numpy as np

from training.shared.dataSplit import temporal_split
from training.shared.graphBuilder import (
    buildCellAdjacencyTyped,
    extractNodeFeatures,
)
from training.shared.normalizer import standardScaler


class GnnDataset:
    """Dataset container for GNN training."""

    def __init__(
        self,
        nodeFeatures: np.ndarray,
        edgeIndex: np.ndarray,
        edgeAttr: np.ndarray | None,
        target: np.ndarray,
        trainIdx: np.ndarray,
        valIdx: np.ndarray,
        testIdx: np.ndarray,
        bcInput: np.ndarray,
        bcScaler: standardScaler,
        featureScaler: standardScaler | None = None,
    ):
        self.nodeFeatures = nodeFeatures
        self.edgeIndex = edgeIndex
        self.edgeAttr = edgeAttr
        self.target = target
        self.trainIdx = trainIdx
        self.valIdx = valIdx
        self.testIdx = testIdx
        self.bcInput = bcInput
        self.bcScaler = bcScaler
        self.featureScaler = featureScaler


def preprocessGnnData(
    snapshotMatrix: np.ndarray,
    schema: dict,
    topologyMeta: dict,
    casePath: str,
    fieldNames: list[str] | None = None,
    targetFieldName: str | str | None = None,
    nodeTypeFields: dict[str, dict] | None = None,
    seqLen: int = 10,
    valRatio: float = 0.2,
    testRatio: float = 0.0,
    seed: int = 42,
    normalize: bool = True,
) -> GnnDataset:
    """
    Build GNN dataset from snapshot matrix.

    Notes on leakage:
    - graph topology is static, built from the whole graph
    - node features are extracted once, but scaling is fit on train only
    - targets are aligned with snapshots, temporal split is applied
    """
    nSnapshots = snapshotMatrix.shape[1]

    # --- Temporal split (no leakage) ---
    trainIdx, valIdx, testIdx = temporal_split(nSnapshots, valRatio, testRatio, seed=seed)

    # --- Node features ---
    # fieldNames -> columns in matrix
    nodeFeatures = extractNodeFeatures(snapshotMatrix, schema, fieldNames)
    # (nSnapshots, nNodes, nNodeFeatures)

    # Feature normalization (fit on train only)
    nSamples, nNodes, nF = nodeFeatures.shape
    if normalize:
        scaler = standardScaler()
        scaler.fit(nodeFeatures[: trainIdx.size].reshape(-1, nF))
        nodeFeatures = scaler.transform(nodeFeatures.reshape(-1, nF)).reshape(nSamples, nNodes, nF)
    else:
        scaler = None

    # --- Graph topology (static, no leakage) ---
    edgeIndex, edgeAttr = buildCellAdjacencyTyped(
        casePath,
        topologyMeta if topologyMeta is not None else None,
    )

    # --- Target / BC ---
    # BC input is extracted from the snapshot matrix using the same logic
    # as for end-to-end / POD, but it may be scaled
    from training.shared.surfaceExtractor import (
        loadFaceOwnerCells,
        loadSchema as loadSchema2,
        loadTopologyMeta as loadTM2,
        filterZones,
        cellIdsForZones,
        zoneMeanFromMatrix,
    )
    faceOwnerCells = loadFaceOwnerCells(casePath)
    nCells = schema["nCells"]
    inletZones = filterZones(topologyMeta, role="boundary", zoneType="inlet")
    wallZones = filterZones(topologyMeta, role="boundary", zoneType="wall")
    inletCellIds = cellIdsForZones(faceOwnerCells, inletZones, maxCellId=nCells)
    wallCellIds = cellIdsForZones(faceOwnerCells, wallZones, maxCellId=nCells)
    bcInput = zoneMeanFromMatrix(snapshotMatrix, schema, inletCellIds, fieldNames or schema.get("fieldOrder", []))

    bcScaler.fit(bcInput[trainIdx])
    bcInputScaled = bcScaler.transform(bcInput)

    target = None
    if targetFieldName is not None:
        target = zoneMeanFromMatrix(snapshotMatrix, schema, wallCellIds, [targetFieldName]).T

    return GnnDataset(
        nodeFeatures=nodeFeatures,
        edgeIndex=edgeIndex,
        edgeAttr=edgeAttr,
        target=target,
        trainIdx=trainIdx,
        valIdx=valIdx,
        testIdx=testIdx,
        bcInput=bcInputScaled,
        bcScaler=bcScaler,
        featureScaler=scaler,
    )


def getGnnTrainLoaders(dataset, batchSize: int):
    """Return train and val DataLoaders for GNN training."""
    import torch
    from torch.utils.data import DataLoader, Subset

    trainDs = Subset(dataset, dataset.trainIdx)
    valDs = Subset(dataset, dataset.valIdx) if dataset.valIdx.size > 0 else None

    trainLoader = DataLoader(trainDs, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valDs, batch_size=batchSize) if valDs is not None else None
    return trainLoader, valLoader


def getGnnTestLoader(dataset, batchSize: int):
    """Return test DataLoader."""
    import torch
    from torch.utils.data import Subset

    testDs = Subset(dataset, dataset.testIdx)
    return DataLoader(testDs, batch_size=batchSize)
