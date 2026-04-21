# preprocess

## step0 inspect h5 structure

`python preprocess/run/inspectH5.py "F:/SIG/dataPreprocess/sampleData/1.dat.h5" "F:/SIG/dataPreprocess/preprocess/out/datInspect.json" "results/1/"`

`python preprocess/run/inspectH5.py "F:/SIG/dataPreprocess/sampleData/1.cas.h5" "F:/SIG/dataPreprocess/preprocess/out/casInspect.json" "meshes/"`

## step1 discover datasets

`python preprocess/run/buildDataset.py discover "F:/SIG/dataPreprocess/sampleData/1.dat.h5" "F:/SIG/dataPreprocess/preprocess/out/datDatasets.json"`

`python preprocess/run/buildDataset.py discover "F:/SIG/dataPreprocess/sampleData/1.cas.h5" "F:/SIG/dataPreprocess/preprocess/out/casDatasets.json"`

## step2 fill field map

fill only project-specific paths in `preprocess/config/extraFieldMap.template.json`.
common fluent fields are preconfigured in `preprocess/config/builtinFieldMap.fluent.json`.

## step3 fill pipeline config

copy `preprocess/config/pipelineConfig.template.json` to `preprocess/config/pipelineConfig.json` and update paths.

## step4 build matrix

`python preprocess/run/buildDataset.py build "F:/SIG/dataPreprocess/preprocess/config/pipelineConfig.json"`

## sample config smoke run

uses `sampleData/1.cas.h5` and `sampleData/1.dat.h5` (fluent-style layout).

`python preprocess/run/buildDataset.py build "F:/SIG/dataPreprocess/preprocess/config/pipelineConfig.sample.json"`

outputs include:
- `snapshotMatrix.npy`
- `metadata.csv`
- `schema.json`
- `topologyMeta.json` (boundary/interface/shadow relations)

## point cell mapping

cfd usually stores cell-centered fields. this module supports both:
- pointData -> cellData (high priority, mesh to cell storage)
- cellData -> pointData (for visualization or coupling)
- interpolation method is pluggable: `mean` or `weighted`

1. run discover on `cas.h5` and fill `preprocess/config/cellToNodeMesh.template.json`:
   - `cellNodeOffsetPaths` / `cellNodeIndexPaths` must point to csr-style cell-node connectivity in your mesh file
   - `nodeCoordPaths` is optional but helps set `nNodes` when ids are zero-based compact
2. python example:

```python
from pathlib import Path
import numpy as np
from preprocess.src.cellToNodeConverter import meshFieldConverter

converter = meshFieldConverter(Path("preprocess/config/cellToNodeMesh.json"))
cellT = converter.pointToCell(Path("case.cas.h5"), pointTemperatureArray, method="mean")
nodeP = converter.cellToPoint(Path("case.cas.h5"), cellPressureArray, method="weighted", cellWeights=cellVolumeArray)
np.save("cellTemperature.npy", cellT)
np.save("nodePressure.npy", nodeP)
```

if fluent stores connectivity under different dataset names, paste the paths from discover output into the json.

## visualization

requires `pyvista`, `vtk`, `matplotlib` (see `requirements.txt`).

mesh node point cloud:

`python preprocess/run/vizRun.py mesh "F:/SIG/dataPreprocess/preprocess/config/vizMesh.sample.json"`

field histogram (and optional cell-centroid point cloud if you provide `cellCentroidsNpy`):

`python preprocess/run/vizRun.py field "F:/SIG/dataPreprocess/preprocess/config/vizField.sample.json"`

full surface/wireframe needs vtk cell connectivity; current mesh view is **node point cloud** only.

## compatibility notes

- phase nodes can be auto-discovered from dat (`results/1/phase-*`) when not explicitly configured.
- dataset leaf compatibility supports candidates without leaf and with `/1..4` leaf suffix.
- mapping now only loads `fieldOrder ∪ requiredFields` to avoid mixed-length SV arrays.
- face zone topology extraction is built-in: boundary map, shadow pairs, and interface pairs.
