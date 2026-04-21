import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from preprocess.src.ioLayer import discoverPhaseNodes
from preprocess.src.topologyLayer import extractTopologyMeta


class testTopologyLayer(unittest.TestCase):
    def testDiscoverPhaseNodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpDir:
            p = Path(tmpDir) / "a.dat.h5"
            with h5py.File(p, "w") as f:
                f.create_dataset("results/1/phase-2/cells/SV_U/1", data=np.array([1.0]))
                f.create_dataset("results/1/phase-1/cells/SV_U/1", data=np.array([1.0]))
            phases = discoverPhaseNodes(p, "results/1")
            self.assertEqual(phases, {"phase1": "phase-1", "phase2": "phase-2"})

    def testExtractTopologyMeta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpDir:
            p = Path(tmpDir) / "a.cas.h5"
            with h5py.File(p, "w") as f:
                z = f.create_group("meshes/1/faces/zoneTopology")
                z.create_dataset("id", data=np.array([10, 11, 12], dtype=np.int32))
                z.create_dataset("minId", data=np.array([1, 5, 8], dtype=np.uint64))
                z.create_dataset("maxId", data=np.array([4, 7, 9], dtype=np.uint64))
                z.create_dataset("zoneType", data=np.array([3, 3, 2], dtype=np.int32))
                z.create_dataset("c0", data=np.array([101, 102, 101], dtype=np.int32))
                z.create_dataset("c1", data=np.array([0, 0, 102], dtype=np.int32))
                z.create_dataset("shadowZoneId", data=np.array([11, 10, 0], dtype=np.int32))
                z.create_dataset("name", data=np.array([b"inlet;inlet-shadow;internal"], dtype="S64"))
            meta = extractTopologyMeta(p)
            self.assertEqual(meta["counts"]["faceZones"], 3)
            self.assertEqual(meta["counts"]["boundaryZones"], 2)
            self.assertEqual(meta["counts"]["shadowPairs"], 1)
            self.assertEqual(meta["counts"]["interfacePairs"], 1)


if __name__ == "__main__":
    unittest.main()
