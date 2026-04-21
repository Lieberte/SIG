import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from preprocess.src.cellToNodeConverter import (
    cellFieldToNodeWeighted,
    cellFieldToNodeMean,
    cellToNodeConverter,
    loadCellNodeIdsFromCsr,
    meshFieldConverter,
    pointFieldToCellMean,
    pointFieldToCellWeighted,
)


class testCellToNodeConverter(unittest.TestCase):
    def testMeanTwoTriangles(self) -> None:
        cellNodeIds = [
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.int64),
        ]
        cellValues = np.array([10.0, 20.0])
        out = cellFieldToNodeMean(cellValues, cellNodeIds, nNodes=4)
        np.testing.assert_allclose(out[0], 10.0)
        np.testing.assert_allclose(out[1], 15.0)
        np.testing.assert_allclose(out[2], 15.0)
        np.testing.assert_allclose(out[3], 20.0)

    def testVectorField(self) -> None:
        cellNodeIds = [
            np.array([0, 1], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
        ]
        cellValues = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = cellFieldToNodeMean(cellValues, cellNodeIds, nNodes=3)
        np.testing.assert_allclose(out[0], [1.0, 2.0])
        np.testing.assert_allclose(out[1], [2.0, 3.0])
        np.testing.assert_allclose(out[2], [3.0, 4.0])
    def testPointToCellMean(self) -> None:
        cellNodeIds = [
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.int64),
        ]
        pointValues = np.array([0.0, 10.0, 20.0, 30.0])
        out = pointFieldToCellMean(pointValues, cellNodeIds)
        np.testing.assert_allclose(out, np.array([10.0, 20.0]))
    def testWeightedMethods(self) -> None:
        cellNodeIds = [
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.int64),
        ]
        cellValues = np.array([10.0, 20.0])
        pointOut = cellFieldToNodeWeighted(
            cellValues=cellValues,
            cellNodeIds=cellNodeIds,
            nNodes=4,
            cellWeights=np.array([1.0, 3.0]),
        )
        np.testing.assert_allclose(pointOut[1], 17.5)
        pointValues = np.array([0.0, 10.0, 20.0, 30.0])
        cellOut = pointFieldToCellWeighted(
            pointValues=pointValues,
            cellNodeIds=cellNodeIds,
            pointWeights=np.array([1.0, 1.0, 2.0, 4.0]),
        )
        np.testing.assert_allclose(cellOut[0], 12.5)
        np.testing.assert_allclose(cellOut[1], 24.285714285714285)

    def testLoadCsrFromH5(self) -> None:
        with tempfile.TemporaryDirectory() as tmpDir:
            path = Path(tmpDir) / "mesh.cas.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("meshes/1/cells/csrOffsets", data=np.array([0, 3, 6], dtype=np.int64))
                f.create_dataset(
                    "meshes/1/cells/csrIndices",
                    data=np.array([0, 1, 2, 1, 2, 3], dtype=np.int64),
                )
            cellNodeIds, nNodes = loadCellNodeIdsFromCsr(
                path,
                ["meshes/1/cells/csrOffsets"],
                ["meshes/1/cells/csrIndices"],
            )
            self.assertEqual(len(cellNodeIds), 2)
            np.testing.assert_array_equal(cellNodeIds[0], [0, 1, 2])
            np.testing.assert_array_equal(cellNodeIds[1], [1, 2, 3])
            self.assertEqual(nNodes, 4)
            values = np.array([10.0, 20.0])
            out = cellFieldToNodeMean(values, cellNodeIds, nNodes=4)
            np.testing.assert_allclose(out[1], 15.0)

    def testConverterClass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpDir:
            tmpPath = Path(tmpDir)
            casePath = tmpPath / "case.cas.h5"
            cfgPath = tmpPath / "mesh.json"
            with h5py.File(casePath, "w") as f:
                f.create_dataset("meshes/1/cells/off", data=np.array([0, 3, 6], dtype=np.int64))
                f.create_dataset("meshes/1/cells/idx", data=np.array([0, 1, 2, 1, 2, 3], dtype=np.int64))
                f.create_dataset("meshes/1/nodes/coords", data=np.zeros((4, 3), dtype=np.float64))
            cfgPath.write_text(
                """{
  "mode": "csr",
  "cellNodeOffsetPaths": ["meshes/1/cells/off"],
  "cellNodeIndexPaths": ["meshes/1/cells/idx"],
  "nodeCoordPaths": ["meshes/1/nodes/coords"],
  "nNodes": null
}""",
                encoding="utf-8",
            )
            converter = cellToNodeConverter(cfgPath)
            out = converter.convertFromCase(casePath, np.array([10.0, 20.0]))
            np.testing.assert_allclose(out[1], 15.0)
    def testMeshFieldConverterPointToCell(self) -> None:
        with tempfile.TemporaryDirectory() as tmpDir:
            tmpPath = Path(tmpDir)
            casePath = tmpPath / "case.cas.h5"
            cfgPath = tmpPath / "mesh.json"
            with h5py.File(casePath, "w") as f:
                f.create_dataset("meshes/1/cells/off", data=np.array([0, 3, 6], dtype=np.int64))
                f.create_dataset("meshes/1/cells/idx", data=np.array([0, 1, 2, 1, 2, 3], dtype=np.int64))
                f.create_dataset("meshes/1/nodes/coords", data=np.zeros((4, 3), dtype=np.float64))
            cfgPath.write_text(
                """{
  "mode": "csr",
  "cellNodeOffsetPaths": ["meshes/1/cells/off"],
  "cellNodeIndexPaths": ["meshes/1/cells/idx"],
  "nodeCoordPaths": ["meshes/1/nodes/coords"],
  "nNodes": null
}""",
                encoding="utf-8",
            )
            converter = meshFieldConverter(cfgPath)
            out = converter.pointToCell(casePath, np.array([0.0, 10.0, 20.0, 30.0]), method="mean")
            np.testing.assert_allclose(out, np.array([10.0, 20.0]))


if __name__ == "__main__":
    unittest.main()
