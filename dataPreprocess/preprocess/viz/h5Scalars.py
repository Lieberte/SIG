from pathlib import Path

import h5py
import numpy as np

from preprocess.src.ioLayer import resolveDataset


def readDataset1d(caseOrDatPath: Path, datasetPath: str) -> np.ndarray:
    with h5py.File(caseOrDatPath, "r") as h5File:
        arr = resolveDataset(h5File, [datasetPath])
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    return flat


def readCoordsNx3(casePath: Path, coordPathCandidates: list[str]) -> np.ndarray:
    with h5py.File(casePath, "r") as h5File:
        coords = resolveDataset(h5File, coordPathCandidates)
    c = np.asarray(coords, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] < 3:
        raise ValueError("coords must be (n, 3) or (n, d) with d>=3")
    return c[:, :3]


def readNpyPoints(path: Path) -> np.ndarray:
    arr = np.load(path)
    c = np.asarray(arr, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] < 3:
        raise ValueError("npy must be (n, 3) cell centers or points")
    return c[:, :3]
