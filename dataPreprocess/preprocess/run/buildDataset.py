import json
import sys
from pathlib import Path


def addProjectRootToPath() -> None:
    currentFile = Path(__file__).resolve()
    projectRoot = currentFile.parents[2]
    if str(projectRoot) not in sys.path:
        sys.path.insert(0, str(projectRoot))


addProjectRootToPath()

from preprocess.src.datasetConverter import datasetConverter


def normalizePath(pathText: str) -> Path:
    return Path(pathText.replace("\\", "/"))


def runPipeline(configPath: Path) -> int:
    converter = datasetConverter(configPath)
    report = converter.runBuild()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def runDiscover(h5Path: Path, outPath: Path) -> int:
    payload = datasetConverter.runDiscover(h5Path, outPath)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage:")
        print("python buildDataset.py discover <h5Path> <outPath>")
        print("python buildDataset.py build <pipelineConfigPath>")
        return 1
    cmd = args[0]
    if cmd == "discover":
        if len(args) != 3:
            return 1
        return runDiscover(normalizePath(args[1]), normalizePath(args[2]))
    if cmd == "build":
        return runPipeline(normalizePath(args[1]))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
