import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class datasetResolveError(Exception):
    def __init__(self, missingItems: list[dict[str, Any]]):
        self.missingItems = missingItems
        lines = ["datasets not found for configured fields:"]
        for item in missingItems:
            lines.append(f"- {item['name']}: {item['candidates']}")
        super().__init__("\n".join(lines))


def loadJson(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mergeDict(baseDict: dict[str, Any], extraDict: dict[str, Any]) -> dict[str, Any]:
    merged = dict(baseDict)
    for key, value in extraDict.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = mergeDict(merged[key], value)
            continue
        if key in merged and isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = list(dict.fromkeys([*merged[key], *value]))
            continue
        merged[key] = value
    return merged


def pathSegments(pathLike: str) -> list[str]:
    return [part for part in pathLike.split("/") if part]


def cleanPath(pathText: str) -> str:
    return "/".join(pathSegments(pathText))


def parseSemicolonNames(rawNames: np.ndarray) -> list[str]:
    flat = np.asarray(rawNames).reshape(-1)
    if flat.size == 0:
        return []
    first = flat[0]
    if isinstance(first, bytes):
        text = first.decode("utf-8", errors="ignore")
    else:
        text = str(first)
    names = [part.strip() for part in text.split(";") if part.strip()]
    return names


def discoverPhaseNodes(dataPath: Path, datPathPrefix: str = "results/1") -> dict[str, str]:
    prefix = cleanPath(datPathPrefix)
    phaseNames: set[str] = set()
    with h5py.File(dataPath, "r") as dataFile:
        if prefix not in dataFile:
            return {}
        root = dataFile[prefix]
        for key in root.keys():
            if str(key).startswith("phase-"):
                phaseNames.add(str(key))
    ordered = sorted(phaseNames, key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 9999)
    output: dict[str, str] = {}
    for idx, node in enumerate(ordered, start=1):
        output[f"phase{idx}"] = node
    return output


def expandLeafCandidates(basePath: str, maxLeaf: int = 4) -> list[str]:
    base = cleanPath(basePath)
    out = [base]
    for i in range(1, maxLeaf + 1):
        out.append(cleanPath(f"{base}/{i}"))
    return list(dict.fromkeys(out))


def generatePhaseFieldPaths(fieldMap: dict[str, Any]) -> dict[str, Any]:
    generated = mergeDict({}, fieldMap)
    fieldPaths = generated.get("fieldPaths", {})
    fieldAliases = generated.get("fieldAliases", {})
    phaseNodes = generated.get("phaseNodes", {})
    datPathPrefix = cleanPath(str(generated.get("datPathPrefix", "results/1")))
    phaseNodeNames = [str(value) for value in phaseNodes.values()]
    for fieldName in fieldAliases:
        if fieldName not in fieldPaths:
            fieldPaths[fieldName] = []
    for fieldName, aliasName in fieldAliases.items():
        if fieldPaths.get(fieldName):
            continue
        if not phaseNodeNames:
            continue
        candidates = []
        for phaseNode in phaseNodeNames:
            base = cleanPath(f"{datPathPrefix}/{phaseNode}/cells/{aliasName}")
            candidates.extend(expandLeafCandidates(base))
        fieldPaths[fieldName] = candidates
    generated["fieldPaths"] = fieldPaths
    return generated


def resolveDatasetOptional(h5File: h5py.File, candidates: list[str]) -> np.ndarray | None:
    for item in candidates:
        nodes = pathSegments(item)
        pointer: Any = h5File
        found = True
        for node in nodes:
            if node not in pointer:
                found = False
                break
            pointer = pointer[node]
        if found:
            return np.asarray(pointer[()])
    return None


def resolveDataset(h5File: h5py.File, candidates: list[str]) -> np.ndarray:
    output = resolveDatasetOptional(h5File, candidates)
    if output is None:
        raise KeyError(f"no dataset found in candidates={candidates}")
    return output


def loadCaseMeta(casePath: Path, indexPaths: dict[str, list[str]]) -> dict[str, np.ndarray]:
    with h5py.File(casePath, "r") as caseFile:
        output: dict[str, np.ndarray] = {}
        for key, candidates in indexPaths.items():
            if key != "casCellIds":
                continue
            if not candidates:
                continue
            output[key] = resolveDataset(caseFile, candidates).reshape(-1)
    return output


def loadDatFrame(
    dataPath: Path,
    fieldMap: dict[str, Any],
    loadFieldNames: set[str] | None = None,
) -> dict[str, Any]:
    fieldPaths = fieldMap.get("fieldPaths", {})
    indexPaths = fieldMap.get("indexPaths", {})
    timePaths = fieldMap.get("timePaths", {})
    requiredFields = set(fieldMap.get("requiredFields", []))
    if loadFieldNames is None:
        loadFieldNames = set(requiredFields)
    frame: dict[str, Any] = {"dataPath": str(dataPath), "fields": {}, "indexes": {}, "timeValue": None}
    missingItems: list[dict[str, Any]] = []
    with h5py.File(dataPath, "r") as dataFile:
        for fieldName, candidates in fieldPaths.items():
            if fieldName not in loadFieldNames:
                continue
            if not candidates:
                if fieldName in requiredFields:
                    missingItems.append({"name": fieldName, "candidates": []})
                continue
            found = resolveDatasetOptional(dataFile, candidates)
            if found is None:
                missingItems.append({"name": fieldName, "candidates": candidates})
                continue
            frame["fields"][fieldName] = found.reshape(-1)
        datIndexCandidates = indexPaths.get("datCellIds", [])
        if datIndexCandidates:
            frame["indexes"]["datCellIds"] = resolveDataset(dataFile, datIndexCandidates).reshape(-1)
        timeCandidates = timePaths.get("timeValue", [])
        if timeCandidates:
            timeArr = resolveDataset(dataFile, timeCandidates).reshape(-1)
            frame["timeValue"] = float(timeArr[0]) if timeArr.size > 0 else None
    if missingItems:
        raise datasetResolveError(missingItems)
    return frame


def discoverDatasets(h5Path: Path) -> list[str]:
    keys: list[str] = []
    with h5py.File(h5Path, "r") as h5File:
        def collect(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                keys.append(name)
        h5File.visititems(collect)
    return keys
