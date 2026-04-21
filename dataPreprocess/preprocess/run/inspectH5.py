import json
import sys
from pathlib import Path
from typing import Any

import h5py


def addProjectRootToPath() -> None:
    currentFile = Path(__file__).resolve()
    projectRoot = currentFile.parents[2]
    if str(projectRoot) not in sys.path:
        sys.path.insert(0, str(projectRoot))


addProjectRootToPath()

from preprocess.src.ioLayer import discoverDatasets


def datasetSummaries(h5Path: Path, prefix: str) -> list[dict]:
    out: list[dict] = []
    with h5py.File(h5Path, "r") as h5File:
        def collect(name: str, obj: h5py.Dataset) -> None:
            if prefix and not name.startswith(prefix):
                return
            item: dict = {"path": name, "shape": list(obj.shape), "dtype": str(obj.dtype)}
            out.append(item)
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                collect(name, obj)
        h5File.visititems(visitor)
    return sorted(out, key=lambda x: x["path"])


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage:")
        print("  python inspectH5.py <h5Path> <outJsonPath> [pathPrefix]")
        print("example:")
        print('  python inspectH5.py "F:/SIG/dataPreprocess/sampleData/1.dat.h5" out/datInspect.json results/1/')
        return 1
    h5Path = Path(args[0].replace("\\", "/"))
    outPath = Path(args[1].replace("\\", "/"))
    prefix = args[2] if len(args) > 2 else ""
    names = discoverDatasets(h5Path)
    summaries = datasetSummaries(h5Path, prefix)
    payload = {
        "h5Path": str(h5Path),
        "pathPrefix": prefix,
        "datasetCount": len(names),
        "datasets": names,
        "datasetSummaries": summaries,
    }
    outPath.parent.mkdir(parents=True, exist_ok=True)
    outPath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"outPath": str(outPath), "datasetCount": len(names), "summaryCount": len(summaries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
