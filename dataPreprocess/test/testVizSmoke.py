import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from preprocess.viz.fieldPlot import renderScalarHistogram
from preprocess.viz.meshPlot import renderNodePointCloud


class testVizSmoke(unittest.TestCase):
    def testHistogram(self) -> None:
        with tempfile.TemporaryDirectory() as tmpDir:
            out = Path(tmpDir) / "h.png"
            renderScalarHistogram(np.random.randn(5000), out, title="t")
            self.assertTrue(out.exists())

    def testPointCloudPng(self) -> None:
        with tempfile.TemporaryDirectory() as tmpDir:
            out = Path(tmpDir) / "m.png"
            coords = np.random.rand(200, 3).astype(np.float64)
            renderNodePointCloud(coords, out, maxPoints=5000)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
