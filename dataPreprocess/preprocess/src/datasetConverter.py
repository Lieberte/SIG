import json
from pathlib import Path

import numpy as np

from preprocess.src.assembleLayer import assembleMatrix, buildSchema
from preprocess.src.ioLayer import (
    discoverDatasets,
    discoverPhaseNodes,
    generatePhaseFieldPaths,
    loadCaseMeta,
    loadDatFrame,
    loadJson,
    mergeDict,
)
from preprocess.src.topologyLayer import extractTopologyMeta
from preprocess.src.transformLayer import alignFrame, fillValueToken


class datasetConverter:
    def __init__(self, configPath: str | Path):
        self.configPath = Path(configPath)
        self.pipelineConfig = loadJson(self.configPath)
    def normalizePath(self, pathText: str) -> Path:
        return Path(pathText.replace("\\", "/"))
    def ensureDir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
    def expandPhaseNodes(self, fieldMap: dict, dataPaths: list[Path]) -> dict:
        phaseNodes = fieldMap.get("phaseNodes", {})
        if phaseNodes:
            return fieldMap
        if not dataPaths:
            return fieldMap
        datPrefix = str(fieldMap.get("datPathPrefix", "results/1"))
        discovered = discoverPhaseNodes(dataPaths[0], datPrefix)
        if not discovered:
            return fieldMap
        newMap = mergeDict({}, fieldMap)
        newMap["phaseNodes"] = discovered
        return newMap
    def loadFieldMap(self) -> dict:
        if "fieldMapPath" in self.pipelineConfig:
            singleMap = loadJson(self.normalizePath(self.pipelineConfig["fieldMapPath"]))
            return generatePhaseFieldPaths(singleMap)
        builtinPath = self.normalizePath(self.pipelineConfig["builtinFieldMapPath"])
        builtinMap = loadJson(builtinPath)
        extraPathText = self.pipelineConfig.get("extraFieldMapPath")
        if not extraPathText:
            return generatePhaseFieldPaths(builtinMap)
        extraPath = self.normalizePath(extraPathText)
        extraMap = loadJson(extraPath)
        merged = mergeDict(builtinMap, extraMap)
        return generatePhaseFieldPaths(merged)
    def runBuild(self) -> dict:
        casePath = self.normalizePath(self.pipelineConfig["casePath"])
        dataPaths = [self.normalizePath(item) for item in self.pipelineConfig["dataPaths"]]
        fieldMap = self.loadFieldMap()
        fieldMap = self.expandPhaseNodes(fieldMap, dataPaths)
        fieldMap = generatePhaseFieldPaths(fieldMap)
        outputDir = self.normalizePath(self.pipelineConfig["outputDir"])
        fieldOrder = self.pipelineConfig["fieldOrder"]
        fillValue = fillValueToken(str(self.pipelineConfig["missingFillValue"]))
        self.ensureDir(outputDir)
        caseMeta = loadCaseMeta(casePath, fieldMap.get("indexPaths", {}))
        loadNames = set(fieldOrder) | set(fieldMap.get("requiredFields", []))
        alignedFrames = []
        for dataPath in dataPaths:
            datFrame = loadDatFrame(dataPath, fieldMap, loadFieldNames=loadNames)
            datFrame["canonicalFromField"] = fieldOrder[0]
            alignedFrames.append(alignFrame(caseMeta, datFrame, fillValue))
        matrix, metadata = assembleMatrix(alignedFrames, fieldOrder)
        schema = buildSchema(alignedFrames, fieldOrder, matrix)
        topologyMeta = extractTopologyMeta(casePath)
        schema["topologyCounts"] = topologyMeta.get("counts", {})
        np.save(outputDir / self.pipelineConfig["matrixFileName"], matrix)
        metadata.to_csv(outputDir / self.pipelineConfig["metadataFileName"], index=False)
        (outputDir / self.pipelineConfig["schemaFileName"]).write_text(
            json.dumps(schema, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        topologyFileName = str(self.pipelineConfig.get("topologyMetaFileName", "topologyMeta.json"))
        (outputDir / topologyFileName).write_text(json.dumps(topologyMeta, ensure_ascii=False, indent=2), encoding="utf-8")
        report = {
            "casePath": str(casePath),
            "dataCount": len(dataPaths),
            "matrixShape": list(matrix.shape),
            "outputDir": str(outputDir),
            "topologyMetaFile": str(outputDir / topologyFileName),
        }
        (outputDir / "runReport.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    @staticmethod
    def runDiscover(h5Path: str | Path, outPath: str | Path) -> dict:
        h5PathObj = Path(h5Path)
        outPathObj = Path(outPath)
        datasets = discoverDatasets(h5PathObj)
        outPathObj.parent.mkdir(parents=True, exist_ok=True)
        payload = {"h5Path": str(h5PathObj), "datasets": datasets}
        outPathObj.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
