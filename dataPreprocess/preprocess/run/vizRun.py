import json
import sys
from pathlib import Path


def addProjectRootToPath() -> None:
    currentFile = Path(__file__).resolve()
    projectRoot = currentFile.parents[2]
    if str(projectRoot) not in sys.path:
        sys.path.insert(0, str(projectRoot))


addProjectRootToPath()

from preprocess.src.ioLayer import loadJson
from preprocess.viz.fieldPlot import runFieldViz
from preprocess.viz.meshPlot import runMeshViz
from preprocess.viz.meshPlotMatplotlib import runMeshVizMatplotlib
from preprocess.viz.surfaceMesh import runSurfacePressureViz, runSurfaceViz


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage:")
        print("  python vizRun.py mesh <configJsonPath>")
        print("  python vizRun.py mesh_mpl <configJsonPath>")
        print("  python vizRun.py surface <configJsonPath>")
        print("  python vizRun.py surface_pressure <configJsonPath>")
        print("  python vizRun.py field <configJsonPath>")
        return 1
    cmd = args[0]
    cfgPath = Path(args[1].replace("\\", "/"))
    config = loadJson(cfgPath)
    if cmd == "mesh":
        report = runMeshViz(config)
    elif cmd == "mesh_mpl":
        report = runMeshVizMatplotlib(config)
    elif cmd == "surface":
        report = runSurfaceViz(config)
    elif cmd == "surface_pressure":
        report = runSurfacePressureViz(config)
    elif cmd == "field":
        report = runFieldViz(config)
    else:
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
