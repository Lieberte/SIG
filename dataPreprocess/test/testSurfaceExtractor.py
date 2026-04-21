import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from training.shared.surfaceExtractor import (
    cellIdsForZones,
    filterZones,
    sliceFieldFromMatrix,
    zoneMeanFromMatrix,
)


def testFilterZonesByRole():
    meta = {
        "faceZones": [
            {"name": "inlet_a", "category": {"role": "boundary", "type": "inlet"}},
            {"name": "wall_top", "category": {"role": "boundary", "type": "wall"}},
            {"name": "interior_0", "category": {"role": "interior", "type": "interior"}},
        ],
    }
    assert len(filterZones(meta, role="boundary")) == 2
    assert len(filterZones(meta, role="boundary", zoneType="inlet")) == 1
    assert len(filterZones(meta, zoneType="wall")) == 1
    assert len(filterZones(meta, role="interior")) == 1


def testCellIdsForZones():
    faceOwnerCells = np.array([3, 5, 7, 1, 2, 10, 8], dtype=np.int64)
    zones = [{"minFaceId": 1, "maxFaceId": 3}]
    ids = cellIdsForZones(faceOwnerCells, zones)
    expected = np.array(sorted({2, 4, 6}), dtype=np.int64)
    np.testing.assert_array_equal(ids, expected)


def testCellIdsForZonesWithMaxCellId():
    faceOwnerCells = np.array([3, 100, 5], dtype=np.int64)
    zones = [{"minFaceId": 1, "maxFaceId": 3}]
    ids = cellIdsForZones(faceOwnerCells, zones, maxCellId=10)
    assert 99 not in ids.tolist()
    assert 2 in ids.tolist()
    assert 4 in ids.tolist()


def testSliceFieldFromMatrix():
    nCells = 5
    nSnapshots = 3
    matrix = np.arange(nCells * 2 * nSnapshots, dtype=np.float64).reshape(
        nCells * 2, nSnapshots,
    )
    schema = {"fieldOrder": ["fieldA", "fieldB"], "nCells": nCells}
    cellIds = np.array([0, 2, 4], dtype=np.int64)
    result = sliceFieldFromMatrix(matrix, schema, "fieldA", cellIds)
    assert result.shape == (3, 3)
    expected = matrix[cellIds, :]
    np.testing.assert_array_equal(result, expected)


def testSliceFieldB():
    nCells = 4
    nSnapshots = 2
    matrix = np.arange(nCells * 2 * nSnapshots, dtype=np.float64).reshape(
        nCells * 2, nSnapshots,
    )
    schema = {"fieldOrder": ["f1", "f2"], "nCells": nCells}
    cellIds = np.array([1, 3], dtype=np.int64)
    result = sliceFieldFromMatrix(matrix, schema, "f2", cellIds)
    assert result.shape == (2, 2)
    expected = matrix[nCells + 1 : nCells * 2 : 2, :]
    np.testing.assert_array_equal(result, expected)


def testZoneMeanFromMatrix():
    nCells = 4
    nSnapshots = 2
    matrix = np.ones((nCells * 2, nSnapshots), dtype=np.float64) * 5.0
    schema = {"fieldOrder": ["f1", "f2"], "nCells": nCells}
    cellIds = np.array([0, 1], dtype=np.int64)
    result = zoneMeanFromMatrix(matrix, schema, cellIds, ["f1"])
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result, 5.0)


if __name__ == "__main__":
    testFilterZonesByRole()
    testCellIdsForZones()
    testCellIdsForZonesWithMaxCellId()
    testSliceFieldFromMatrix()
    testSliceFieldB()
    testZoneMeanFromMatrix()
    print("all surfaceExtractor tests passed")
