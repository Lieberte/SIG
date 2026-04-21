import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from preprocess.src.datasetConverter import datasetConverter


class testDatasetConverter(unittest.TestCase):
    def testRunDiscover(self) -> None:
        with tempfile.TemporaryDirectory() as tmpDir:
            tmpPath = Path(tmpDir)
            h5Path = tmpPath / "a.h5"
            outPath = tmpPath / "out" / "datasets.json"
            with h5py.File(h5Path, "w") as f:
                f.create_dataset("x/y", data=np.array([1, 2]))
            payload = datasetConverter.runDiscover(h5Path, outPath)
            self.assertEqual(payload["h5Path"], str(h5Path))
            saved = json.loads(outPath.read_text(encoding="utf-8"))
            self.assertEqual(saved["datasets"], ["x/y"])
    def testLoadFieldMapMerge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpDir:
            tmpPath = Path(tmpDir)
            builtinPath = tmpPath / "builtin.json"
            extraPath = tmpPath / "extra.json"
            pipelinePath = tmpPath / "pipeline.json"
            builtinPath.write_text(
                json.dumps(
                    {
                        "requiredFields": ["velocityX"],
                        "fieldPaths": {"velocityX": ["a/x"]},
                        "indexPaths": {"datCellIds": ["a/id"]},
                    }
                ),
                encoding="utf-8",
            )
            extraPath.write_text(
                json.dumps(
                    {
                        "requiredFields": ["pressure"],
                        "fieldPaths": {"pressure": ["b/p"], "velocityX": ["b/u"]},
                        "indexPaths": {"casCellIds": ["b/cid"]},
                    }
                ),
                encoding="utf-8",
            )
            pipelinePath.write_text(
                json.dumps(
                    {
                        "builtinFieldMapPath": str(builtinPath),
                        "extraFieldMapPath": str(extraPath),
                        "casePath": "x",
                        "dataPaths": [],
                        "outputDir": "x",
                        "fieldOrder": [],
                        "missingFillValue": "nan",
                        "matrixFileName": "a.npy",
                        "metadataFileName": "b.csv",
                        "schemaFileName": "c.json",
                    }
                ),
                encoding="utf-8",
            )
            converter = datasetConverter(pipelinePath)
            merged = converter.loadFieldMap()
            self.assertIn("velocityX", merged["requiredFields"])
            self.assertIn("pressure", merged["requiredFields"])
            self.assertEqual(merged["fieldPaths"]["pressure"], ["b/p"])
            self.assertEqual(merged["fieldPaths"]["velocityX"], ["a/x", "b/u"])
            self.assertEqual(merged["indexPaths"]["datCellIds"], ["a/id"])
            self.assertEqual(merged["indexPaths"]["casCellIds"], ["b/cid"])
    def testGeneratePhaseAutoPath(self) -> None:
        with tempfile.TemporaryDirectory() as tmpDir:
            tmpPath = Path(tmpDir)
            builtinPath = tmpPath / "builtin.json"
            pipelinePath = tmpPath / "pipeline.json"
            builtinPath.write_text(
                json.dumps(
                    {
                        "datPathPrefix": "results/1",
                        "phaseNodes": {"mixture": "phase-1", "gas": "phase-2"},
                        "fieldAliases": {"velocityX": "SV_U"},
                        "fieldPaths": {"velocityX": []},
                    }
                ),
                encoding="utf-8",
            )
            pipelinePath.write_text(
                json.dumps(
                    {
                        "fieldMapPath": str(builtinPath),
                        "casePath": "x",
                        "dataPaths": [],
                        "outputDir": "x",
                        "fieldOrder": [],
                        "missingFillValue": "nan",
                        "matrixFileName": "a.npy",
                        "metadataFileName": "b.csv",
                        "schemaFileName": "c.json",
                    }
                ),
                encoding="utf-8",
            )
            converter = datasetConverter(pipelinePath)
            merged = converter.loadFieldMap()
            candidates = merged["fieldPaths"]["velocityX"]
            self.assertIn("results/1/phase-1/cells/SV_U/1", candidates)
            self.assertIn("results/1/phase-2/cells/SV_U/1", candidates)


if __name__ == "__main__":
    unittest.main()
