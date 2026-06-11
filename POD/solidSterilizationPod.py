"""POD model for F0 sterilization prediction on fluid-soild interface faces."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from solidTemperaturePod import (
    parseCaseTemperatures, parseTime, discoverCaseDirs,
    normalizeParams, fitPod, makeFeatures,
    fitCoefficientRegression, rbfKernel, fitRbfRegression, fitIdwRegression,
    fitRegression,
    predictLocalRbf, predictIdw, predictCoefficients, reconstruct, computeMetrics,
    trainPod, evaluatePod, evaluateProjection,
    splitTrainTest, getSplitCaseNames,
    plotCloudComparison, plotThreePointComparison, saveModel,
)


def _selectPointIdx(coords: np.ndarray | None, nSolid: int, nPoints: int) -> np.ndarray:
    nPoints = max(1, int(nPoints))
    if coords is None or len(coords) != nSolid:
        return np.unique(np.linspace(0, nSolid - 1, min(nPoints, nSolid), dtype=np.int64))
    x = coords[:, 0]; y = coords[:, 1]; z = coords[:, 2]
    center = coords.mean(axis=0)
    candidates = [
        int(np.argmin(x)), int(np.argmin(np.linalg.norm(coords - center, axis=1))),
        int(np.argmax(x)), int(np.argmin(y)), int(np.argmax(y)),
        int(np.argmin(z)), int(np.argmax(z)),
    ]
    for q in np.linspace(0.1, 0.9, max(nPoints, 3)):
        candidates.append(int(np.argmin(np.abs(x - np.quantile(x, q)))))
    selected = []; seen = set()
    for idx in candidates:
        if idx not in seen:
            selected.append(idx); seen.add(idx)
        if len(selected) >= nPoints:
            break
    return np.asarray(selected, dtype=np.int64)


def _selectCloudSnapshots(testIdx: np.ndarray, metadata: list[dict],
                           casesPerRun: int, snapshotsPerCase: int) -> list[tuple[str, int, int]]:
    pairs = []; usedCases = 0
    for case in metadata:
        localPos = np.where((testIdx >= case["snapshotStart"]) & (testIdx < case["snapshotEnd"]))[0]
        if len(localPos) == 0:
            continue
        if casesPerRun > 0 and usedCases >= casesPerRun:
            break
        nTake = max(1, min(int(snapshotsPerCase), len(localPos)))
        samplePos = np.unique(np.linspace(0, len(localPos) - 1, nTake, dtype=np.int64))
        for pos in localPos[samplePos]:
            pairs.append((case["caseName"], int(pos), int(testIdx[pos])))
        usedCases += 1
    return pairs


def _solidCellRange(casPath: Path) -> tuple[int, int]:
    with h5py.File(casPath, "r") as f:
        cz = f["meshes/1/cells/zoneTopology"]
        mins = np.asarray(cz["minId"][()]).reshape(-1)
        maxs = np.asarray(cz["maxId"][()]).reshape(-1)
        namesRaw = np.asarray(cz["name"][()]).reshape(-1)
        if namesRaw.size > 0:
            first = namesRaw[0]
            text = first.decode("utf-8", errors="ignore") if isinstance(first, bytes) else str(first)
            names = [p.strip() for p in text.split(";") if p.strip()]
        else:
            names = []
        for i, name in enumerate(names):
            if name == "soild":
                return int(mins[i]) - 1, int(maxs[i] - mins[i] + 1)
    raise ValueError("Cannot find solid cell zone in CAS mesh")


def _faceZoneRange(casPath: Path, zoneName: str) -> tuple[int, int]:
    with h5py.File(casPath, "r") as f:
        z = f["meshes/1/faces/zoneTopology"]
        mins = np.asarray(z["minId"][()]).reshape(-1)
        maxs = np.asarray(z["maxId"][()]).reshape(-1)
        namesRaw = np.asarray(z["name"][()]).reshape(-1)
        if namesRaw.size > 0:
            first = namesRaw[0]
            text = first.decode("utf-8", errors="ignore") if isinstance(first, bytes) else str(first)
            names = [p.strip() for p in text.split(";") if p.strip()]
        else:
            names = []
        order = np.argsort(mins)
        offset = 0
        for i in order:
            count = int(maxs[i] - mins[i] + 1)
            if names[i] == zoneName:
                return offset, count
            offset += count
    raise ValueError(f"Cannot find face zone '{zoneName}' in CAS mesh")


def _faceCoords(casPath: Path) -> np.ndarray:
    """Compute face center coordinates for fluid_i-soild faces."""
    faceStart, faceCount = _faceZoneRange(casPath, "fluid_i-soild")
    with h5py.File(casPath, "r") as f:
        nodeCoords = np.asarray(f["meshes/1/nodes/coords/1"][()])
        faceNN = np.asarray(f["meshes/1/faces/nodes/1/nnodes"][()])
        faceNodes = np.asarray(f["meshes/1/faces/nodes/1/nodes"][()])
    offsets = np.cumsum(np.concatenate([[0], faceNN]))
    centers = np.zeros((faceCount, 3), dtype=np.float64)
    for localIdx in range(faceCount):
        globalFaceId = faceStart + localIdx
        nids = faceNodes[offsets[globalFaceId]:offsets[globalFaceId + 1]]
        centers[localIdx] = nodeCoords[nids - 1].mean(axis=0)
    return centers


def _detectDataFormat(datPath: Path) -> str:
    with h5py.File(datPath, "r") as f:
        return "1phase" if "results/1/phase-2" not in f else "3phase"


def readSterilizationF0(datPath: Path, udmColumn: int, casPath: Path) -> np.ndarray:
    fmt = _detectDataFormat(datPath)
    if fmt != "1phase":
        raise ValueError(f"F0 sterilization requires 1-phase format, got {fmt}")
    faceStart, faceCount = _faceZoneRange(casPath, "fluid_i-soild")
    h5path = "results/1/phase-1/faces/SV_UDM_I/1"
    with h5py.File(datPath, "r") as file:
        faceUdm = np.asarray(file[h5path][()], dtype=np.float64)
    return faceUdm[faceStart:faceStart + faceCount, int(udmColumn)]


def loadSterilizationSnapshots(dataRoot: Path, caseGlob: str, datGlob: str,
                                udmColumn: int, casPath: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    caseDirs = discoverCaseDirs(dataRoot, caseGlob, datGlob)
    if not caseDirs:
        raise FileNotFoundError(f"No case folders under {dataRoot} matching {caseGlob}/{datGlob}")
    faceStart, faceCount = _faceZoneRange(casPath, "fluid_i-soild")
    snapshots = []
    params = []
    metadata = []
    nFaces = None
    for casePath in caseDirs:
        tPreheatK, tH2O2K = parseCaseTemperatures(casePath)
        datFiles = sorted(casePath.glob(datGlob), key=parseTime)
        caseTimes = [parseTime(datPath) for datPath in datFiles]
        caseStart = len(snapshots)
        for datPath, timeValue in zip(datFiles, caseTimes):
            f0 = readSterilizationF0(datPath, udmColumn, casPath)
            if nFaces is None:
                nFaces = f0.size
            if f0.size != nFaces:
                raise ValueError(f"{datPath} face count {f0.size} differs from {nFaces}")
            snapshots.append(f0)
            params.append([timeValue, tPreheatK, tH2O2K])
        metadata.append({
            "caseName": casePath.name,
            "casePath": str(casePath),
            "tPreheatK": tPreheatK,
            "tH2O2K": tH2O2K,
            "nSnapshots": len(caseTimes),
            "timeMin": float(min(caseTimes)) if caseTimes else None,
            "timeMax": float(max(caseTimes)) if caseTimes else None,
            "times": caseTimes,
            "snapshotStart": caseStart,
            "snapshotEnd": len(snapshots),
        })
    X = np.asarray(snapshots, dtype=np.float64).T
    P = np.asarray(params, dtype=np.float64)
    return X, P, metadata, {"faceStart": faceStart, "faceCount": faceCount}


def main() -> None:
    _projectRoot = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="POD model for F0 sterilization on fluid_i-soild faces")
    parser.add_argument("--dataRoot", type=Path, default=_projectRoot / "data2")
    parser.add_argument("--caseGlob", default="T_*_*")
    parser.add_argument("--datGlob", default="*.dat.h5")
    parser.add_argument("--casPath", type=Path, default=_projectRoot / "data" / "meshData" / "KMB_phase_change_test.cas.h5")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output" / "solidSterilizationPod.npz")
    parser.add_argument("--figDir", type=Path, default=Path(__file__).resolve().parent / "output" / "figures")
    parser.add_argument("--energy", type=float, default=0.999)
    parser.add_argument("--maxModes", type=int, default=0)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--regressionType", choices=["poly", "rbf", "idw", "mlp"], default="mlp")
    parser.add_argument("--rbfEpsilon", type=float, default=1.0)
    parser.add_argument("--rbfRidge", type=float, default=1e-5)
    parser.add_argument("--rbfNeighbors", type=int, default=64)
    parser.add_argument("--idwNeighbors", type=int, default=64)
    parser.add_argument("--idwPower", type=float, default=2.0)
    parser.add_argument("--idwEps", type=float, default=1e-12)
    parser.add_argument("--mlpHidden", type=int, default=128)
    parser.add_argument("--mlpLayers", type=int, default=3)
    parser.add_argument("--mlpEpochs", type=int, default=3000)
    parser.add_argument("--mlpLearningRate", type=float, default=1e-3)
    parser.add_argument("--mlpWeightDecay", type=float, default=1e-6)
    parser.add_argument("--mlpValRatio", type=float, default=0.1)
    parser.add_argument("--mlpPatience", type=int, default=300)
    parser.add_argument("--udmColumn", type=int, default=2)
    parser.add_argument("--testRatio", type=float, default=0.2)
    parser.add_argument("--splitMode", choices=["temporal", "random", "case"], default="temporal")
    parser.add_argument("--testCases", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxPlotPoints", type=int, default=12000)
    parser.add_argument("--nComparePoints", type=int, default=6)
    parser.add_argument("--cloudCasesPerRun", type=int, default=0)
    parser.add_argument("--cloudSnapshotsPerCase", type=int, default=3)
    args = parser.parse_args()

    maxModes = args.maxModes if args.maxModes > 0 else None
    X, P, metadata, faceInfo = loadSterilizationSnapshots(
        args.dataRoot, args.caseGlob, args.datGlob, args.udmColumn, args.casPath)

    testCaseNames = [name.strip() for name in args.testCases.split(",") if name.strip()]
    trainIdx, testIdx = splitTrainTest(metadata, X.shape[1], args.testRatio,
                                        args.splitMode, args.seed, testCaseNames)
    if len(trainIdx) == 0 or len(testIdx) == 0:
        raise ValueError(f"Invalid split: train={len(trainIdx)} test={len(testIdx)}")
    trainCaseNames = getSplitCaseNames(metadata, trainIdx)
    splitTestCaseNames = getSplitCaseNames(metadata, testIdx)

    model = trainPod(
        X[:, trainIdx], P[trainIdx],
        args.energy, maxModes,
        args.degree, args.ridge, args.regressionType,
        args.rbfEpsilon, args.rbfRidge, args.rbfNeighbors,
        args.idwNeighbors, args.idwPower, args.idwEps,
        args.mlpHidden, args.mlpLayers, args.mlpEpochs,
        args.mlpLearningRate, args.mlpWeightDecay, args.mlpValRatio, args.mlpPatience,
    )

    _, trainMetrics = evaluatePod(model, X[:, trainIdx], P[trainIdx])
    testPred, testMetrics = evaluatePod(model, X[:, testIdx], P[testIdx])
    _, trainProjectionMetrics = evaluateProjection(model, X[:, trainIdx])
    _, testProjectionMetrics = evaluateProjection(model, X[:, testIdx])

    faceCoords = _faceCoords(args.casPath)
    pointIdx = _selectPointIdx(faceCoords, X.shape[0], args.nComparePoints)

    cloudFigures = []
    for caseName, cloudSnapshotLocal, cloudSnapshotGlobal in _selectCloudSnapshots(
        testIdx, metadata, args.cloudCasesPerRun, args.cloudSnapshotsPerCase):
        cloudPath = args.figDir / f"cloudCompare_F0_{caseName}_snapshot_{cloudSnapshotGlobal:04d}.png"
        plotCloudComparison(
            faceCoords, X[:, cloudSnapshotGlobal], testPred[:, cloudSnapshotLocal],
            cloudPath, args.maxPlotPoints, args.seed + cloudSnapshotGlobal,
        )
        cloudFigures.append(str(cloudPath))

    plotThreePointComparison(
        P, X, testPred, testIdx, pointIdx, metadata,
        args.figDir / "threePointCompare_F0.png",
        (float(P[:, 0].min()), float(P[:, 0].max())),
    )

    summaryExtra = {
        "fieldType": "F0",
        "udmColumn": args.udmColumn,
        "faceZone": "fluid_i-soild",
        "faceStart": faceInfo["faceStart"],
        "faceCount": faceInfo["faceCount"],
        "splitMode": args.splitMode,
        "testRatio": args.testRatio,
        "regressionType": args.regressionType,
        "rbfEpsilon": args.rbfEpsilon, "rbfRidge": args.rbfRidge, "rbfNeighbors": args.rbfNeighbors,
        "idwNeighbors": args.idwNeighbors, "idwPower": args.idwPower, "idwEps": args.idwEps,
        "mlpHidden": args.mlpHidden, "mlpLayers": args.mlpLayers,
        "mlpEpochs": args.mlpEpochs, "mlpLearningRate": args.mlpLearningRate,
        "mlpWeightDecay": args.mlpWeightDecay, "mlpValRatio": args.mlpValRatio,
        "mlpPatience": args.mlpPatience,
        "nComparePoints": args.nComparePoints,
        "cloudCasesPerRun": args.cloudCasesPerRun,
        "cloudSnapshotsPerCase": args.cloudSnapshotsPerCase,
        "trainCases": trainCaseNames,
        "testCases": splitTestCaseNames,
        "trainIdx": trainIdx.tolist(), "testIdx": testIdx.tolist(),
        "trainMetrics": trainMetrics, "testMetrics": testMetrics,
        "trainProjectionMetrics": trainProjectionMetrics,
        "testProjectionMetrics": testProjectionMetrics,
        "pointIdx": pointIdx.tolist(),
        "figures": {
            "cloudCompare": cloudFigures,
            "threePointCompare": str(args.figDir / "threePointCompare_F0.png"),
        },
    }
    saveModel(model, args.output, metadata, P, summaryExtra)
    print(json.dumps({
        "output": str(args.output),
        "figDir": str(args.figDir),
        "fieldType": "F0",
        "faceZone": "fluid_i-soild",
        "nFaces": int(X.shape[0]),
        "nSnapshots": int(X.shape[1]),
        "nTrain": int(len(trainIdx)),
        "nTest": int(len(testIdx)),
        "splitMode": args.splitMode,
        "regressionType": args.regressionType,
        "rbfEpsilon": args.rbfEpsilon, "rbfRidge": args.rbfRidge, "rbfNeighbors": args.rbfNeighbors,
        "idwNeighbors": args.idwNeighbors, "idwPower": args.idwPower, "idwEps": args.idwEps,
        "mlpHidden": args.mlpHidden, "mlpLayers": args.mlpLayers,
        "mlpEpochs": args.mlpEpochs, "mlpLearningRate": args.mlpLearningRate,
        "mlpWeightDecay": args.mlpWeightDecay, "mlpValRatio": args.mlpValRatio,
        "mlpPatience": args.mlpPatience,
        "nComparePoints": args.nComparePoints,
        "cloudCasesPerRun": args.cloudCasesPerRun,
        "cloudSnapshotsPerCase": args.cloudSnapshotsPerCase,
        "trainCases": trainCaseNames, "testCases": splitTestCaseNames,
        "nModes": int(model["pod"]["nModes"]),
        "energy": float(model["pod"]["energyRatio"][model["pod"]["nModes"] - 1]),
        "trainMetrics": trainMetrics, "testMetrics": testMetrics,
        "trainProjectionMetrics": trainProjectionMetrics,
        "testProjectionMetrics": testProjectionMetrics,
        "pointIdx": pointIdx.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
