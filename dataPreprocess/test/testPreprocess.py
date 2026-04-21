import unittest
import tempfile
from pathlib import Path

import h5py
import numpy as np

from preprocess.src.assembleLayer import assembleMatrix, buildSnapshot
from preprocess.src.ioLayer import datasetResolveError, loadDatFrame
from preprocess.src.transformLayer import alignFrame, fillValueToken


class testTransformLayer(unittest.TestCase):
    def testAlignFrameByCellId(self) -> None:
        caseMeta = {"casCellIds": np.array([10, 20, 30], dtype=np.int64)}
        datFrame = {
            "dataPath": "sample.dat.h5",
            "timeValue": 1.0,
            "indexes": {"datCellIds": np.array([20, 30, 10], dtype=np.int64)},
            "fields": {"pressure": np.array([2.0, 3.0, 1.0])},
        }
        aligned = alignFrame(caseMeta, datFrame, fillValue=float("nan"))
        np.testing.assert_array_equal(aligned["canonicalCellIds"], np.array([10, 20, 30], dtype=np.int64))
        np.testing.assert_allclose(aligned["fields"]["pressure"], np.array([1.0, 2.0, 3.0]))

    def testFillValueToken(self) -> None:
        self.assertTrue(np.isnan(fillValueToken("nan")))
        self.assertEqual(fillValueToken("0.0"), 0.0)


class testAssembleLayer(unittest.TestCase):
    def testBuildSnapshotAndMatrix(self) -> None:
        frameA = {
            "canonicalCellIds": np.array([1, 2], dtype=np.int64),
            "timeValue": 0.1,
            "dataPath": "a.dat.h5",
            "fields": {"velocityX": np.array([1.0, 2.0]), "pressure": np.array([3.0, 4.0])},
        }
        frameB = {
            "canonicalCellIds": np.array([1, 2], dtype=np.int64),
            "timeValue": 0.2,
            "dataPath": "b.dat.h5",
            "fields": {"velocityX": np.array([10.0, 20.0]), "pressure": np.array([30.0, 40.0])},
        }
        snapshot = buildSnapshot(frameA, ["velocityX", "pressure"])
        np.testing.assert_allclose(snapshot, np.array([1.0, 2.0, 3.0, 4.0]))
        matrix, metadata = assembleMatrix([frameA, frameB], ["velocityX", "pressure"])
        self.assertEqual(matrix.shape, (4, 2))
        np.testing.assert_allclose(matrix[:, 0], np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(matrix[:, 1], np.array([10.0, 20.0, 30.0, 40.0]))
        self.assertEqual(metadata.shape[0], 2)


class testIoLayer(unittest.TestCase):
    def testCollectAllMissingFields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpDir:
            tmpPath = Path(tmpDir)
            datPath = tmpPath / "sample.dat.h5"
            with h5py.File(datPath, "w") as f:
                f.create_dataset("results/1/phase-1/cells/SV_U/1", data=np.array([1.0, 2.0]))
            fieldMap = {
                "requiredFields": ["velocityX", "pressure", "temperature"],
                "fieldPaths": {
                    "velocityX": ["results/1/phase-1/cells/SV_U/1"],
                    "pressure": ["results/1/phase-1/cells/SV_P/1"],
                    "temperature": ["results/1/phase-1/cells/SV_T/1"],
                },
                "indexPaths": {},
                "timePaths": {},
            }
            with self.assertRaises(datasetResolveError) as ctx:
                loadDatFrame(datPath, fieldMap)
            missingNames = [item["name"] for item in ctx.exception.missingItems]
            self.assertEqual(set(missingNames), {"pressure", "temperature"})


if __name__ == "__main__":
    unittest.main()
