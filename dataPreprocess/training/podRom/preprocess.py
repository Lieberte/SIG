"""
POD-ROM specific data preprocessing.

Handles:
- Temporal split with no leakage
- POD fit only on training data
- BC extraction with enhanced statistics
- Energy analysis and reporting
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from training.shared.dataSplit import temporal_split, DataSplitter
from training.shared.podBasis import podBasis
from training.shared.normalizer import standardScaler
from training.shared.surfaceExtractor import (
    cellIdsForZones,
    filterZones,
    loadFaceOwnerCells,
    loadSchema,
    loadTopologyMeta,
    sliceFieldFromMatrix,
    zoneMeanFromMatrix,
)


@dataclass
class BcStatistics:
    """Enhanced boundary condition statistics."""

    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    median: np.ndarray
    q25: np.ndarray
    q75: np.ndarray

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "median": self.median.tolist(),
            "q25": self.q25.tolist(),
            "q75": self.q75.tolist(),
        }


@dataclass
class PodRomDataset:
    """Container for POD-ROM training data."""

    # POD basis (fit on train only)
    pod: podBasis
    # Latent coefficients: (nSnapshots, nModes)
    latentCoefficients: np.ndarray
    # BC input: (nSnapshots, nBcFeatures)
    bcInput: np.ndarray
    # Target field on surface: (nSnapshots, nSurfaceCells)
    surfaceTarget: np.ndarray | None
    # Split indices
    trainIdx: np.ndarray
    valIdx: np.ndarray
    testIdx: np.ndarray
    # Normalizers
    bcScaler: standardScaler
    latentScaler: standardScaler
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def getTrainData(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get training data."""
        latent = self.latentCoefficients[self.trainIdx]
        bc = self.bcInput[self.trainIdx]
        target = self.surfaceTarget[self.trainIdx] if self.surfaceTarget is not None else None
        return latent, bc, target

    def getValData(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get validation data."""
        latent = self.latentCoefficients[self.valIdx]
        bc = self.bcInput[self.valIdx]
        target = self.surfaceTarget[self.valIdx] if self.surfaceTarget is not None else None
        return latent, bc, target

    def getTestData(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get test data."""
        latent = self.latentCoefficients[self.testIdx]
        bc = self.bcInput[self.testIdx]
        target = self.surfaceTarget[self.testIdx] if self.surfaceTarget is not None else None
        return latent, bc, target

    def save(self, outputDir: Path) -> None:
        """Save dataset to disk."""
        outputDir = Path(outputDir)
        outputDir.mkdir(parents=True, exist_ok=True)

        self.pod.save(outputDir / "pod")

        np.save(outputDir / "latentCoefficients.npy", self.latentCoefficients)
        np.save(outputDir / "bcInput.npy", self.bcInput)
        if self.surfaceTarget is not None:
            np.save(outputDir / "surfaceTarget.npy", self.surfaceTarget)

        np.save(outputDir / "trainIdx.npy", self.trainIdx)
        np.save(outputDir / "valIdx.npy", self.valIdx)
        np.save(outputDir / "testIdx.npy", self.testIdx)

        np.savez(
            outputDir / "scalers.npz",
            bcMean=self.bcScaler.mean,
            bcStd=self.bcScaler.std,
            latentMean=self.latentScaler.mean,
            latentStd=self.latentScaler.std,
        )

        meta = {
            **self.metadata,
            "nTrain": len(self.trainIdx),
            "nVal": len(self.valIdx),
            "nTest": len(self.testIdx),
            "nModes": self.pod.nModes,
        }
        (outputDir / "datasetMeta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    @classmethod
    def load(cls, inputDir: Path) -> "PodRomDataset":
        """Load dataset from disk."""
        inputDir = Path(inputDir)

        pod = podBasis.load(inputDir / "pod")
        latentCoefficients = np.load(inputDir / "latentCoefficients.npy")
        bcInput = np.load(inputDir / "bcInput.npy")
        surfaceTarget = np.load(inputDir / "surfaceTarget.npy") if (inputDir / "surfaceTarget.npy").exists() else None

        trainIdx = np.load(inputDir / "trainIdx.npy")
        valIdx = np.load(inputDir / "valIdx.npy")
        testIdx = np.load(inputDir / "testIdx.npy")

        scalers = np.load(inputDir / "scalers.npz")
        bcScaler = standardScaler()
        bcScaler.mean = scalers["bcMean"]
        bcScaler.std = scalers["bcStd"]
        latentScaler = standardScaler()
        latentScaler.mean = scalers["latentMean"]
        latentScaler.std = scalers["latentStd"]

        meta = {}
        if (inputDir / "datasetMeta.json").exists():
            meta = json.loads((inputDir / "datasetMeta.json").read_text(encoding="utf-8"))

        return cls(
            pod=pod,
            latentCoefficients=latentCoefficients,
            bcInput=bcInput,
            surfaceTarget=surfaceTarget,
            trainIdx=trainIdx,
            valIdx=valIdx,
            testIdx=testIdx,
            bcScaler=bcScaler,
            latentScaler=latentScaler,
            metadata=meta,
        )


def computeBcStatistics(
    matrix: np.ndarray,
    schema: dict,
    cellIds: np.ndarray,
    fieldNames: list[str],
) -> BcStatistics:
    """Compute enhanced BC statistics for a zone."""
    nSnapshots = matrix.shape[1]
    data = np.zeros((nSnapshots, len(fieldNames)), dtype=np.float64)

    for i, name in enumerate(fieldNames):
        values = sliceFieldFromMatrix(matrix, schema, name, cellIds)
        data[:, i] = np.nanmean(values, axis=0)

    return BcStatistics(
        mean=np.nanmean(data, axis=0),
        std=np.nanstd(data, axis=0),
        min=np.nanmin(data, axis=0),
        max=np.nanmax(data, axis=0),
        median=np.nanmedian(data, axis=0),
        q25=np.nanpercentile(data, 25, axis=0),
        q75=np.nanpercentile(data, 75, axis=0),
    )


def extractBcWithStats(
    matrix: np.ndarray,
    schema: dict,
    topologyMeta: dict,
    casePath: Path,
    bcFieldNames: list[str],
    targetFieldName: str | None = None,
    inletRole: str = "boundary",
    inletType: str = "inlet",
    wallRole: str = "boundary",
    wallType: str = "wall",
) -> dict[str, Any]:
    """Extract BC input and target with enhanced statistics."""
    faceOwnerCells = loadFaceOwnerCells(casePath)
    nCells = schema["nCells"]

    inletZones = filterZones(topologyMeta, role=inletRole, zoneType=inletType)
    wallZones = filterZones(topologyMeta, role=wallRole, zoneType=wallType)

    inletCellIds = cellIdsForZones(faceOwnerCells, inletZones, maxCellId=nCells)
    wallCellIds = cellIdsForZones(faceOwnerCells, wallZones, maxCellId=nCells)

    # BC input: zone mean
    bcInput = zoneMeanFromMatrix(matrix, schema, inletCellIds, bcFieldNames)

    # BC statistics
    bcStats = computeBcStatistics(matrix, schema, inletCellIds, bcFieldNames)

    # Target field on wall surface
    surfaceTarget = None
    if targetFieldName is not None:
        surfaceTarget = sliceFieldFromMatrix(matrix, schema, targetFieldName, wallCellIds).T

    return {
        "bcInput": bcInput,
        "bcStats": bcStats,
        "surfaceTarget": surfaceTarget,
        "inletCellIds": inletCellIds,
        "wallCellIds": wallCellIds,
        "nInletCells": len(inletCellIds),
        "nWallCells": len(wallCellIds),
    }


class PodRomPreprocessor:
    """
    Preprocessor for POD-ROM training pipeline.

    Handles:
    - Temporal split (no leakage)
    - POD fit on train data only
    - BC extraction with statistics
    - Energy analysis
    """

    def __init__(
        self,
        nModes: int = 20,
        valRatio: float = 0.2,
        testRatio: float = 0.0,
        normalize: bool = True,
    ):
        self.nModes = nModes
        self.valRatio = valRatio
        self.testRatio = testRatio
        self.normalize = normalize

    def prepare(
        self,
        snapshotMatrix: np.ndarray,
        schema: dict,
        topologyMeta: dict,
        casePath: Path,
        bcFieldNames: list[str],
        targetFieldName: str | None = None,
    ) -> PodRomDataset:
        """
        Prepare POD-ROM dataset.

        Args:
            snapshotMatrix: Field snapshot matrix (nCells*nFields, nSnapshots)
            schema: Schema dict from preprocessing
            topologyMeta: Topology metadata
            casePath: Path to case file
            bcFieldNames: Field names for BC input
            targetFieldName: Target field name (optional)

        Returns:
            PodRomDataset ready for training
        """
        nSnapshots = snapshotMatrix.shape[1]

        # Temporal split (no leakage)
        trainIdx, valIdx, testIdx = temporal_split(
            nSnapshots, self.valRatio, self.testRatio
        )

        # POD fit on TRAIN data only
        trainMatrix = snapshotMatrix[:, trainIdx]
        pod = podBasis(self.nModes)
        pod.fit(trainMatrix)

        # Encode all snapshots using train-fitted POD
        latentCoefficients = pod.encode(snapshotMatrix).T  # (nSnapshots, nModes)

        # Extract BC and target
        bcData = extractBcWithStats(
            snapshotMatrix, schema, topologyMeta, casePath,
            bcFieldNames, targetFieldName,
        )
        bcInput = bcData["bcInput"]
        surfaceTarget = bcData["surfaceTarget"]

        # Normalizers fit on train data
        bcScaler = standardScaler()
        latentScaler = standardScaler()

        if self.normalize:
            bcInput = bcScaler.fitTransform(bcInput[trainIdx])
            bcInput_full = bcScaler.transform(bcData["bcInput"])
            bcInput = bcInput_full

            latentTrain = latentCoefficients[trainIdx]
            latentScaler.fit(latentTrain)
            latentCoefficients = latentScaler.transform(latentCoefficients)

        metadata = {
            "nSnapshots": nSnapshots,
            "nModes": self.nModes,
            "valRatio": self.valRatio,
            "testRatio": self.testRatio,
            "normalize": self.normalize,
            "bcFieldNames": bcFieldNames,
            "targetFieldName": targetFieldName,
            "podEnergyReport": pod.energyReport(),
            "bcStats": bcData["bcStats"].to_dict(),
            "nInletCells": bcData["nInletCells"],
            "nWallCells": bcData["nWallCells"],
        }

        return PodRomDataset(
            pod=pod,
            latentCoefficients=latentCoefficients,
            bcInput=bcInput,
            surfaceTarget=surfaceTarget,
            trainIdx=trainIdx,
            valIdx=valIdx,
            testIdx=testIdx,
            bcScaler=bcScaler,
            latentScaler=latentScaler,
            metadata=metadata,
        )
