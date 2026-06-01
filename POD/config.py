from dataclasses import dataclass
from pathlib import Path


@dataclass
class PodConfig:
    projectRoot: Path = Path(__file__).resolve().parents[1]
    dataRoot: Path = projectRoot / "data"
    caseGlob: str = "T_*_*"
    datGlob: str = "*.dat.h5"
    casPath: Path = projectRoot / "data" / "meshData" / "KMB_phase_change_test.cas.h5"
    outputPath: Path = Path(__file__).resolve().parent / "output" / "solidTemperaturePod.npz"
    figDir: Path = Path(__file__).resolve().parent / "output" / "figures"
    energy: float = 0.999
    maxModes: int = 0
    degree: int = 2
    ridge: float = 1e-8
    regressionType: str = "idw"
    rbfEpsilon: float = 1.0
    rbfRidge: float = 1e-5
    rbfNeighbors: int = 64
    idwNeighbors: int = 64
    idwPower: float = 2.0
    idwEps: float = 1e-12
    testRatio: float = 0.2
    splitMode: str = "case"
    testCases: str = ""
    seed: int = 42
    maxPlotPoints: int = 12000
