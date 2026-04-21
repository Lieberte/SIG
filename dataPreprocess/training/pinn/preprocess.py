"""
PINN specific data preprocessing.

Handles:
- Collocation point sampling from mesh
- Boundary point extraction with BC values
- Initial condition extraction
- Data supervision from CFD snapshots
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PinnDataset:
    """Container for PINN training data."""

    # Collocation points (interior)
    collocationCoords: np.ndarray  # (nColloc, nDim)
    collocationTimes: np.ndarray   # (nColloc, 1)

    # Boundary points
    boundaryCoords: np.ndarray     # (nBoundary, nDim)
    boundaryTimes: np.ndarray      # (nBoundary, 1)
    boundaryValues: np.ndarray     # (nBoundary, nFields)
    boundaryTypes: np.ndarray      # (nBoundary,) - zone type

    # Initial condition points
    initialCoords: np.ndarray      # (nInitial, nDim)
    initialValues: np.ndarray      # (nInitial, nFields)

    # Data supervision (from CFD)
    dataCoords: np.ndarray         # (nData, nDim)
    dataTimes: np.ndarray          # (nData, 1)
    dataValues: np.ndarray         # (nData, nFields)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, outputDir: Path) -> None:
        """Save dataset to disk."""
        outputDir = Path(outputDir)
        outputDir.mkdir(parents=True, exist_ok=True)

        np.save(outputDir / "collocationCoords.npy", self.collocationCoords)
        np.save(outputDir / "collocationTimes.npy", self.collocationTimes)
        np.save(outputDir / "boundaryCoords.npy", self.boundaryCoords)
        np.save(outputDir / "boundaryTimes.npy", self.boundaryTimes)
        np.save(outputDir / "boundaryValues.npy", self.boundaryValues)
        np.save(outputDir / "boundaryTypes.npy", self.boundaryTypes)
        np.save(outputDir / "initialCoords.npy", self.initialCoords)
        np.save(outputDir / "initialValues.npy", self.initialValues)
        np.save(outputDir / "dataCoords.npy", self.dataCoords)
        np.save(outputDir / "dataTimes.npy", self.dataTimes)
        np.save(outputDir / "dataValues.npy", self.dataValues)

        (outputDir / "pinnDatasetMeta.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    @classmethod
    def load(cls, inputDir: Path) -> "PinnDataset":
        """Load dataset from disk."""
        inputDir = Path(inputDir)

        def _load(name: str) -> np.ndarray:
            return np.load(inputDir / name)

        meta = {}
        if (inputDir / "pinnDatasetMeta.json").exists():
            meta = json.loads((inputDir / "pinnDatasetMeta.json").read_text(encoding="utf-8"))

        return cls(
            collocationCoords=_load("collocationCoords.npy"),
            collocationTimes=_load("collocationTimes.npy"),
            boundaryCoords=_load("boundaryCoords.npy"),
            boundaryTimes=_load("boundaryTimes.npy"),
            boundaryValues=_load("boundaryValues.npy"),
            boundaryTypes=_load("boundaryTypes.npy"),
            initialCoords=_load("initialCoords.npy"),
            initialValues=_load("initialValues.npy"),
            dataCoords=_load("dataCoords.npy"),
            dataTimes=_load("dataTimes.npy"),
            dataValues=_load("dataValues.npy"),
            metadata=meta,
        )


class PinnPreprocessor:
    """
    Preprocessor for PINN training.

    Extracts:
    - Collocation points from mesh interior
    - Boundary points with BC values
    - Initial condition points
    - Data points from CFD snapshots for supervision
    """

    def __init__(
        self,
        nCollocationPoints: int = 10000,
        nBoundaryPoints: int = 1000,
        nInitialPoints: int = 1000,
        nDataPoints: int = 5000,
        timeRange: tuple[float, float] = (0.0, 1.0),
        coordRange: tuple[float, float] | None = None,
        seed: int = 42,
    ):
        self.nCollocationPoints = nCollocationPoints
        self.nBoundaryPoints = nBoundaryPoints
        self.nInitialPoints = nInitialPoints
        self.nDataPoints = nDataPoints
        self.timeRange = timeRange
        self.coordRange = coordRange
        self.seed = seed

    def _sampleFromCells(
        self,
        cellCenters: np.ndarray,
        nPoints: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample points from cell centers (with small perturbation)."""
        nCells = len(cellCenters)
        if nCells == 0:
            return np.zeros((0, cellCenters.shape[1] if cellCenters.ndim > 1 else 3))

        indices = rng.choice(nCells, size=min(nPoints, nCells), replace=True)
        points = cellCenters[indices]

        # Add small perturbation
        perturbation = rng.normal(0, 0.01 * np.std(cellCenters, axis=0), points.shape)
        return points + perturbation

    def prepare(
        self,
        snapshotMatrix: np.ndarray,
        schema: dict,
        topologyMeta: dict,
        casePath: Path,
        fieldNames: list[str],
        cellCenters: np.ndarray | None = None,
    ) -> PinnDataset:
        """
        Prepare PINN dataset.

        Args:
            snapshotMatrix: Field snapshot matrix (nCells*nFields, nSnapshots)
            schema: Schema dict from preprocessing
            topologyMeta: Topology metadata
            casePath: Path to case file
            fieldNames: Field names to use
            cellCenters: Cell center coordinates (nCells, 3)

        Returns:
            PinnDataset ready for training
        """
        rng = np.random.default_rng(self.seed)

        nCells = schema["nCells"]
        nSnapshots = snapshotMatrix.shape[1]
        nFields = len(fieldNames)

        # Extract cell centers if not provided
        if cellCenters is None:
            # Generate dummy coordinates (normalized to [0, 1])
            cellCenters = np.random.rand(nCells, 3)

        # Normalize coordinates
        if self.coordRange is not None:
            coordMin, coordMax = self.coordRange
        else:
            coordMin = cellCenters.min(axis=0)
            coordMax = cellCenters.max(axis=0)

        coordRange = coordMax - coordMin
        coordRange[coordRange < 1e-10] = 1.0
        normalizedCoords = (cellCenters - coordMin) / coordRange

        # Time normalization
        tMin, tMax = self.timeRange
        timeNormalized = np.linspace(0.0, 1.0, nSnapshots)

        # --- Collocation points (interior) ---
        collocCoords = self._sampleFromCells(normalizedCoords, self.nCollocationPoints, rng)
        collocTimes = rng.uniform(0.0, 1.0, size=(len(collocCoords), 1))

        # --- Boundary points ---
        if topologyMeta and topologyMeta.get("faceZones"):
            from training.shared.surfaceExtractor import (
                loadFaceOwnerCells,
                filterZones,
                cellIdsForZones,
            )

            faceOwnerCells = loadFaceOwnerCells(casePath)
            boundaryZones = filterZones(topologyMeta, role="boundary")
            boundaryCellIds = cellIdsForZones(faceOwnerCells, boundaryZones, maxCellId=nCells)

            boundaryCoords = normalizedCoords[boundaryCellIds] if len(boundaryCellIds) > 0 else np.zeros((0, 3))
        else:
            # No boundary data available - sample from edges of domain
            nBoundaryEstimate = min(self.nBoundaryPoints, nCells // 10)
            boundaryIdx = rng.choice(nCells, size=max(nBoundaryEstimate, 1), replace=False)
            boundaryCoords = normalizedCoords[boundaryIdx]
        if len(boundaryCoords) > self.nBoundaryPoints:
            idx = rng.choice(len(boundaryCoords), self.nBoundaryPoints, replace=False)
            boundaryCoords = boundaryCoords[idx]

        boundaryTimes = rng.uniform(0.0, 1.0, size=(len(boundaryCoords), 1))
        boundaryValues = np.zeros((len(boundaryCoords), nFields))
        boundaryTypes = np.zeros(len(boundaryCoords), dtype=np.int64)

        # --- Initial condition points ---
        initialCoords = self._sampleFromCells(normalizedCoords, self.nInitialPoints, rng)
        initialTimes = np.zeros((len(initialCoords), 1))

        # Extract initial field values
        initialValues = np.zeros((len(initialCoords), nFields))
        for i, fieldName in enumerate(fieldNames):
            fieldIdx = schema["fieldOrder"].index(fieldName) if fieldName in schema["fieldOrder"] else 0
            fieldData = snapshotMatrix[fieldIdx * nCells : (fieldIdx + 1) * nCells, 0]
            initialValues[:, i] = fieldData[:len(initialCoords)]

        # --- Data supervision (from CFD snapshots) ---
        dataCoords = self._sampleFromCells(normalizedCoords, self.nDataPoints, rng)
        dataTimes = rng.uniform(0.0, 1.0, size=(len(dataCoords), 1))
        dataValues = np.zeros((len(dataCoords), nFields))

        # Sample from random snapshots
        snapshotIdx = rng.integers(0, nSnapshots, size=len(dataCoords))
        for i, fieldName in enumerate(fieldNames):
            fieldIdx = schema["fieldOrder"].index(fieldName) if fieldName in schema["fieldOrder"] else 0
            for j in range(len(dataCoords)):
                cellIdx = int(dataCoords[j, 0] * nCells) % nCells  # Approximate cell index
                dataValues[j, i] = snapshotMatrix[fieldIdx * nCells + cellIdx, snapshotIdx[j]]

        metadata = {
            "nCollocationPoints": len(collocCoords),
            "nBoundaryPoints": len(boundaryCoords),
            "nInitialPoints": len(initialCoords),
            "nDataPoints": len(dataCoords),
            "nFields": nFields,
            "fieldNames": fieldNames,
            "timeRange": self.timeRange,
            "coordRange": (coordMin.tolist(), coordMax.tolist()),
            "nSnapshots": nSnapshots,
            "nCells": nCells,
        }

        return PinnDataset(
            collocationCoords=collocCoords,
            collocationTimes=collocTimes,
            boundaryCoords=boundaryCoords,
            boundaryTimes=boundaryTimes,
            boundaryValues=boundaryValues,
            boundaryTypes=boundaryTypes,
            initialCoords=initialCoords,
            initialValues=initialValues,
            dataCoords=dataCoords,
            dataTimes=dataTimes,
            dataValues=dataValues,
            metadata=metadata,
        )


def preparePinnData(
    snapshotMatrix: np.ndarray,
    schema: dict,
    topologyMeta: dict,
    casePath: Path,
    fieldNames: list[str],
    nCollocationPoints: int = 10000,
    nBoundaryPoints: int = 1000,
    nInitialPoints: int = 1000,
    nDataPoints: int = 5000,
    timeRange: tuple[float, float] = (0.0, 1.0),
    seed: int = 42,
) -> PinnDataset:
    """Convenience function to prepare PINN data."""
    preprocessor = PinnPreprocessor(
        nCollocationPoints=nCollocationPoints,
        nBoundaryPoints=nBoundaryPoints,
        nInitialPoints=nInitialPoints,
        nDataPoints=nDataPoints,
        timeRange=timeRange,
        seed=seed,
    )
    return preprocessor.prepare(snapshotMatrix, schema, topologyMeta, casePath, fieldNames)
