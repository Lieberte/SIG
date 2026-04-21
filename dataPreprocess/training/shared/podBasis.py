import json
from pathlib import Path

import numpy as np


class podBasis:
    def __init__(self, nModes: int = 10):
        self.nModes = nModes
        self.mean: np.ndarray | None = None
        self.modes: np.ndarray | None = None
        self.singularValues: np.ndarray | None = None
        self.allSingularValues: np.ndarray | None = None

    def fit(self, snapshotMatrix: np.ndarray) -> "podBasis":
        self.mean = snapshotMatrix.mean(axis=1, keepdims=True)
        centered = snapshotMatrix - self.mean
        U, s, _Vt = np.linalg.svd(centered, full_matrices=False)
        r = min(self.nModes, U.shape[1])
        self.modes = U[:, :r]
        self.singularValues = s[:r]
        self.allSingularValues = s
        self.nModes = r
        return self

    def encode(self, snapshotMatrix: np.ndarray) -> np.ndarray:
        centered = snapshotMatrix - self.mean
        return self.modes.T @ centered

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        return self.modes @ coefficients + self.mean

    def cumulativeEnergy(self) -> np.ndarray:
        if self.allSingularValues is None:
            return np.array([])
        total = np.sum(self.allSingularValues ** 2)
        if total < 1e-30:
            return np.zeros_like(self.allSingularValues)
        return np.cumsum(self.allSingularValues ** 2) / total

    def truncatedEnergy(self) -> float:
        if self.singularValues is None or self.allSingularValues is None:
            return 0.0
        total = np.sum(self.allSingularValues ** 2)
        if total < 1e-30:
            return 0.0
        return float(np.sum(self.singularValues ** 2) / total)

    def reconstructionError(self, snapshotMatrix: np.ndarray) -> float:
        latent = self.encode(snapshotMatrix)
        reconstructed = self.decode(latent)
        residual = np.linalg.norm(snapshotMatrix - reconstructed)
        original = np.linalg.norm(snapshotMatrix)
        if original < 1e-30:
            return 0.0
        return float(residual / original)

    def energyReport(self) -> dict:
        """Generate a detailed energy analysis report."""
        if self.allSingularValues is None:
            return {}
        cumEnergy = self.cumulativeEnergy()
        totalEnergy = float(np.sum(self.allSingularValues ** 2))
        report = {
            "nModesRequested": self.nModes,
            "nModesUsed": self.modes.shape[1] if self.modes is not None else 0,
            "totalEnergy": totalEnergy,
            "truncatedEnergy": self.truncatedEnergy(),
            "singularValues": [float(s) for s in self.singularValues],
            "cumulativeEnergyAtK": {
                str(k + 1): float(cumEnergy[k])
                for k in range(min(50, len(cumEnergy)))
            },
            "modesFor99": int(np.searchsorted(cumEnergy, 0.99) + 1) if len(cumEnergy) > 0 else 0,
            "modesFor95": int(np.searchsorted(cumEnergy, 0.95) + 1) if len(cumEnergy) > 0 else 0,
            "modesFor90": int(np.searchsorted(cumEnergy, 0.90) + 1) if len(cumEnergy) > 0 else 0,
        }
        return report

    def perModeContribution(self) -> np.ndarray:
        """Return fractional energy contribution per mode."""
        if self.allSingularValues is None:
            return np.array([])
        total = np.sum(self.allSingularValues ** 2)
        if total < 1e-30:
            return np.zeros_like(self.allSingularValues)
        return (self.allSingularValues ** 2) / total

    def save(self, outputDir: Path) -> None:
        """Persist POD basis to disk."""
        outputDir = Path(outputDir)
        outputDir.mkdir(parents=True, exist_ok=True)
        if self.mean is not None:
            np.save(outputDir / "podMean.npy", self.mean)
        if self.modes is not None:
            np.save(outputDir / "podModes.npy", self.modes)
        if self.singularValues is not None:
            np.save(outputDir / "podSingularValues.npy", self.singularValues)
        if self.allSingularValues is not None:
            np.save(outputDir / "podAllSingularValues.npy", self.allSingularValues)
        report = self.energyReport()
        (outputDir / "podEnergyReport.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )

    @classmethod
    def load(cls, inputDir: Path) -> "podBasis":
        """Load POD basis from disk."""
        inputDir = Path(inputDir)
        obj = cls()
        if (inputDir / "podMean.npy").exists():
            obj.mean = np.load(inputDir / "podMean.npy")
        if (inputDir / "podModes.npy").exists():
            obj.modes = np.load(inputDir / "podModes.npy")
            obj.nModes = obj.modes.shape[1]
        if (inputDir / "podSingularValues.npy").exists():
            obj.singularValues = np.load(inputDir / "podSingularValues.npy")
        if (inputDir / "podAllSingularValues.npy").exists():
            obj.allSingularValues = np.load(inputDir / "podAllSingularValues.npy")
        return obj
        if original < 1e-30:
            return 0.0
        return float(residual / original)
