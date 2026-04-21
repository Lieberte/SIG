import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from training.shared.podBasis import podBasis


def testFitEncodeDecode():
    np.random.seed(42)
    nFeatures, nSnapshots = 50, 10
    data = np.random.randn(nFeatures, nSnapshots)
    pod = podBasis(nModes=5)
    pod.fit(data)
    assert pod.modes.shape == (nFeatures, 5)
    assert pod.singularValues.shape == (5,)
    latent = pod.encode(data)
    assert latent.shape == (5, nSnapshots)
    reconstructed = pod.decode(latent)
    assert reconstructed.shape == (nFeatures, nSnapshots)


def testReconstructionError():
    np.random.seed(42)
    nFeatures, nSnapshots = 30, 20
    base = np.random.randn(nFeatures, 3)
    coeffs = np.random.randn(3, nSnapshots)
    data = base @ coeffs + np.random.randn(nFeatures, nSnapshots) * 0.01
    pod = podBasis(nModes=3).fit(data)
    err = pod.reconstructionError(data)
    assert err < 0.05


def testCumulativeEnergy():
    np.random.seed(42)
    data = np.random.randn(20, 10)
    pod = podBasis(nModes=5).fit(data)
    energy = pod.cumulativeEnergy()
    assert energy.shape[0] == pod.allSingularValues.shape[0]
    assert energy[-1] > 0.99


def testTruncatedEnergy():
    np.random.seed(42)
    data = np.random.randn(20, 10)
    pod = podBasis(nModes=10).fit(data)
    assert abs(pod.truncatedEnergy() - 1.0) < 1e-10


def testSingleSnapshot():
    data = np.random.randn(10, 1)
    pod = podBasis(nModes=3).fit(data)
    assert pod.nModes == 1
    latent = pod.encode(data)
    assert latent.shape == (1, 1)


if __name__ == "__main__":
    testFitEncodeDecode()
    testReconstructionError()
    testCumulativeEnergy()
    testTruncatedEnergy()
    testSingleSnapshot()
    print("all podBasis tests passed")
