import argparse
import json
import re
from pathlib import Path

import numpy as np

from config import PodConfig

repoRoot = Path(__file__).resolve().parents[1]


def parseCaseTemperatures(casePath: Path) -> tuple[float, float]:
    match = re.search(r"T_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)", casePath.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse temperatures from {casePath.name}")
    return float(match.group(1)) + 273.15, float(match.group(2)) + 273.15


def parseTime(datPath: Path) -> float:
    text = datPath.stem.split("-")[-1].replace(".dat", "")
    return float(text)


def discoverCaseDirs(dataRoot: Path, caseGlob: str, datGlob: str) -> list[Path]:
    return sorted(path for path in dataRoot.glob(caseGlob) if path.is_dir() and any(path.glob(datGlob)))


def readSolidTemperature(datPath: Path) -> np.ndarray:
    import h5py
    with h5py.File(datPath, "r") as file:
        path = "results/1/phase-1/cells/SV_T/1"
        if path not in file:
            raise KeyError(f"{datPath} missing {path}")
        return np.asarray(file[path][()], dtype=np.float64).reshape(-1)


def loadSolidSnapshots(dataRoot: Path, caseGlob: str, datGlob: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    caseDirs = discoverCaseDirs(dataRoot, caseGlob, datGlob)
    if not caseDirs:
        raise FileNotFoundError(f"No case folders under {dataRoot} matching {caseGlob}/{datGlob}")
    snapshots = []
    params = []
    metadata = []
    nSolid = None
    for casePath in caseDirs:
        tPreheatK, tH2O2K = parseCaseTemperatures(casePath)
        datFiles = sorted(casePath.glob(datGlob), key=parseTime)
        caseTimes = [parseTime(datPath) for datPath in datFiles]
        caseStart = len(snapshots)
        for datPath, timeValue in zip(datFiles, caseTimes):
            solidT = readSolidTemperature(datPath)
            if nSolid is None:
                nSolid = solidT.size
            if solidT.size != nSolid:
                raise ValueError(f"{datPath} solid size {solidT.size} differs from {nSolid}")
            snapshots.append(solidT)
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
    return X, P, metadata


def normalizeParams(P: np.ndarray) -> tuple[np.ndarray, dict]:
    pMin = P.min(axis=0)
    pMax = P.max(axis=0)
    pRange = np.maximum(pMax - pMin, 1e-12)
    PN = (P - pMin) / pRange
    return PN, {"min": pMin, "range": pRange}


def fitPod(X: np.ndarray, energy: float, maxModes: int | None) -> dict:
    mean = X.mean(axis=1, keepdims=True)
    XCentered = X - mean
    U, S, _ = np.linalg.svd(XCentered, full_matrices=False)
    energyRatio = np.cumsum(S ** 2) / np.sum(S ** 2)
    nModes = int(np.searchsorted(energyRatio, energy) + 1)
    if maxModes is not None:
        nModes = min(nModes, maxModes)
    PHI = U[:, :nModes]
    A = PHI.T @ XCentered
    return {"mean": mean, "PHI": PHI, "A": A, "S": S, "energyRatio": energyRatio, "nModes": nModes}


def makeFeatures(PN: np.ndarray, degree: int) -> np.ndarray:
    t = PN[:, 0]
    tp = PN[:, 1]
    th = PN[:, 2]
    features = [np.ones_like(t), t, tp, th]
    if degree >= 2:
        features.extend([t * t, tp * tp, th * th, t * tp, t * th, tp * th])
    if degree >= 3:
        features.extend([t ** 3, tp ** 3, th ** 3, t * tp * th])
    return np.vstack(features).T


def fitCoefficientRegression(P: np.ndarray, A: np.ndarray, degree: int, ridge: float) -> dict:
    PN, paramScale = normalizeParams(P)
    F = makeFeatures(PN, degree)
    lhs = F.T @ F + ridge * np.eye(F.shape[1])
    rhs = F.T @ A.T
    W = np.linalg.solve(lhs, rhs)
    return {"type": "poly", "W": W, "degree": degree, "ridge": ridge, "paramScale": paramScale}


def rbfKernel(PA: np.ndarray, PB: np.ndarray, epsilon: float) -> np.ndarray:
    diff = PA[:, None, :] - PB[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    return np.exp(-((epsilon ** 2) * dist2))


def fitRbfRegression(P: np.ndarray, A: np.ndarray, ridge: float, epsilon: float, neighbors: int) -> dict:
    PN, paramScale = normalizeParams(P)
    return {
        "type": "rbf",
        "ATrain": A.T,
        "PTrainNorm": PN,
        "ridge": ridge,
        "epsilon": epsilon,
        "neighbors": neighbors,
        "paramScale": paramScale,
    }


def fitIdwRegression(P: np.ndarray, A: np.ndarray, neighbors: int, power: float, eps: float) -> dict:
    PN, paramScale = normalizeParams(P)
    return {
        "type": "idw",
        "ATrain": A.T,
        "PTrainNorm": PN,
        "neighbors": neighbors,
        "power": power,
        "eps": eps,
        "paramScale": paramScale,
    }


def buildMlp(inputDim: int, outputDim: int, hidden: int, layers: int):
    import torch.nn as nn
    modules = []
    lastDim = inputDim
    for _ in range(max(1, int(layers))):
        modules.extend([nn.Linear(lastDim, hidden), nn.Tanh()])
        lastDim = hidden
    modules.append(nn.Linear(lastDim, outputDim))
    return nn.Sequential(*modules)


def fitMlpRegression(
    P: np.ndarray,
    A: np.ndarray,
    hidden: int,
    layers: int,
    epochs: int,
    learningRate: float,
    weightDecay: float,
    valRatio: float,
    patience: int,
) -> dict:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("POD-NN requires torch.") from exc
    PN, paramScale = normalizeParams(P)
    Y = A.T.astype(np.float32)
    yMean = Y.mean(axis=0, keepdims=True)
    yStd = Y.std(axis=0, keepdims=True) + 1e-8
    YN = (Y - yMean) / yStd
    nSamples = PN.shape[0]
    rng = np.random.default_rng(42)
    order = rng.permutation(nSamples)
    nVal = int(round(nSamples * valRatio)) if nSamples > 10 else 0
    nVal = min(max(nVal, 0), nSamples - 1) if nSamples > 1 else 0
    valIdx = order[:nVal]
    trainIdx = order[nVal:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = buildMlp(PN.shape[1], YN.shape[1], int(hidden), int(layers)).to(device)
    xTrain = torch.tensor(PN[trainIdx], dtype=torch.float32, device=device)
    yTrain = torch.tensor(YN[trainIdx], dtype=torch.float32, device=device)
    xVal = torch.tensor(PN[valIdx], dtype=torch.float32, device=device) if nVal > 0 else None
    yVal = torch.tensor(YN[valIdx], dtype=torch.float32, device=device) if nVal > 0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate, weight_decay=weightDecay)
    lossFn = torch.nn.MSELoss()
    bestLoss = float("inf")
    bestState = None
    noImprove = 0
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        loss = lossFn(model(xTrain), yTrain)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            checkLoss = lossFn(model(xVal), yVal).item() if nVal > 0 else loss.item()
        if checkLoss < bestLoss:
            bestLoss = checkLoss
            bestState = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            noImprove = 0
        else:
            noImprove += 1
        if patience > 0 and noImprove >= patience:
            break
    if bestState is not None:
        model.load_state_dict(bestState)
    model.eval()
    return {
        "type": "mlp",
        "model": model.cpu(),
        "hidden": int(hidden),
        "layers": int(layers),
        "epochs": int(epochs),
        "learningRate": float(learningRate),
        "weightDecay": float(weightDecay),
        "valRatio": float(valRatio),
        "patience": int(patience),
        "bestLoss": float(bestLoss),
        "coeffMean": yMean.astype(np.float64),
        "coeffStd": yStd.astype(np.float64),
        "paramScale": paramScale,
    }


def fitRegression(
    P: np.ndarray,
    A: np.ndarray,
    regressionType: str,
    degree: int,
    ridge: float,
    rbfEpsilon: float,
    rbfRidge: float,
    rbfNeighbors: int,
    idwNeighbors: int,
    idwPower: float,
    idwEps: float,
    mlpHidden: int,
    mlpLayers: int,
    mlpEpochs: int,
    mlpLearningRate: float,
    mlpWeightDecay: float,
    mlpValRatio: float,
    mlpPatience: int,
) -> dict:
    if regressionType == "poly":
        return fitCoefficientRegression(P, A, degree, ridge)
    if regressionType == "rbf":
        return fitRbfRegression(P, A, rbfRidge, rbfEpsilon, rbfNeighbors)
    if regressionType == "idw":
        return fitIdwRegression(P, A, idwNeighbors, idwPower, idwEps)
    if regressionType == "mlp":
        return fitMlpRegression(P, A, mlpHidden, mlpLayers, mlpEpochs, mlpLearningRate, mlpWeightDecay, mlpValRatio, mlpPatience)
    raise ValueError(f"Unsupported regressionType: {regressionType}")


def predictLocalRbf(PN: np.ndarray, regression: dict) -> np.ndarray:
    PTrain = regression["PTrainNorm"]
    ATrain = regression["ATrain"]
    epsilon = regression["epsilon"]
    ridge = regression["ridge"]
    neighbors = int(regression["neighbors"])
    nTrain = PTrain.shape[0]
    k = nTrain if neighbors <= 0 else min(neighbors, nTrain)
    coeff = np.zeros((PN.shape[0], ATrain.shape[1]), dtype=np.float64)
    for i, point in enumerate(PN):
        dist2 = np.sum((PTrain - point) ** 2, axis=1)
        if k < nTrain:
            idx = np.argpartition(dist2, k - 1)[:k]
        else:
            idx = np.arange(nTrain)
        idx = idx[np.argsort(dist2[idx])]
        K = rbfKernel(PTrain[idx], PTrain[idx], epsilon)
        lhs = K + ridge * np.eye(len(idx))
        alpha = np.linalg.solve(lhs, ATrain[idx])
        kQuery = rbfKernel(point.reshape(1, -1), PTrain[idx], epsilon)
        coeff[i] = kQuery @ alpha
    return coeff.T


def predictIdw(PN: np.ndarray, regression: dict) -> np.ndarray:
    PTrain = regression["PTrainNorm"]
    ATrain = regression["ATrain"]
    neighbors = int(regression["neighbors"])
    power = float(regression["power"])
    eps = float(regression["eps"])
    nTrain = PTrain.shape[0]
    k = nTrain if neighbors <= 0 else min(neighbors, nTrain)
    coeff = np.zeros((PN.shape[0], ATrain.shape[1]), dtype=np.float64)
    for i, point in enumerate(PN):
        dist = np.sqrt(np.sum((PTrain - point) ** 2, axis=1))
        if k < nTrain:
            idx = np.argpartition(dist, k - 1)[:k]
        else:
            idx = np.arange(nTrain)
        idx = idx[np.argsort(dist[idx])]
        if dist[idx[0]] <= eps:
            coeff[i] = ATrain[idx[0]]
            continue
        weights = 1.0 / np.maximum(dist[idx], eps) ** power
        weights = weights / np.sum(weights)
        coeff[i] = weights @ ATrain[idx]
    return coeff.T


def predictCoefficients(P: np.ndarray, regression: dict) -> np.ndarray:
    pMin = regression["paramScale"]["min"]
    pRange = regression["paramScale"]["range"]
    PN = (P - pMin) / pRange
    if regression["type"] == "poly":
        F = makeFeatures(PN, regression["degree"])
        return (F @ regression["W"]).T
    if regression["type"] == "rbf":
        return predictLocalRbf(PN, regression)
    if regression["type"] == "idw":
        return predictIdw(PN, regression)
    if regression["type"] == "mlp":
        import torch
        model = regression["model"]
        model.eval()
        with torch.no_grad():
            yNorm = model(torch.tensor(PN, dtype=torch.float32)).cpu().numpy()
        y = yNorm * regression["coeffStd"] + regression["coeffMean"]
        return y.T
    raise ValueError(f"Unsupported regression type: {regression['type']}")


def reconstruct(pod: dict, A: np.ndarray) -> np.ndarray:
    return pod["mean"] + pod["PHI"] @ A


def computeMetrics(Y: np.ndarray, YPred: np.ndarray) -> dict:
    err = YPred - Y
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    relRmse = float(rmse / (np.std(Y) + 1e-12))
    return {
        "mseK2": mse,
        "rmseK": rmse,
        "maeK": mae,
        "relativeRmse": relRmse,
        "actualMinK": float(np.min(Y)),
        "actualMaxK": float(np.max(Y)),
        "predMinK": float(np.min(YPred)),
        "predMaxK": float(np.max(YPred)),
        "errorMinK": float(np.min(err)),
        "errorMaxK": float(np.max(err)),
        "absErrorMaxK": float(np.max(np.abs(err))),
    }


def trainPod(
    X: np.ndarray,
    P: np.ndarray,
    energy: float,
    maxModes: int | None,
    degree: int,
    ridge: float,
    regressionType: str,
    rbfEpsilon: float,
    rbfRidge: float,
    rbfNeighbors: int,
    idwNeighbors: int,
    idwPower: float,
    idwEps: float,
    mlpHidden: int,
    mlpLayers: int,
    mlpEpochs: int,
    mlpLearningRate: float,
    mlpWeightDecay: float,
    mlpValRatio: float,
    mlpPatience: int,
) -> dict:
    pod = fitPod(X, energy, maxModes)
    regression = fitRegression(
        P,
        pod["A"],
        regressionType,
        degree,
        ridge,
        rbfEpsilon,
        rbfRidge,
        rbfNeighbors,
        idwNeighbors,
        idwPower,
        idwEps,
        mlpHidden,
        mlpLayers,
        mlpEpochs,
        mlpLearningRate,
        mlpWeightDecay,
        mlpValRatio,
        mlpPatience,
    )
    APred = predictCoefficients(P, regression)
    XPred = reconstruct(pod, APred)
    metrics = computeMetrics(X, XPred)
    return {"pod": pod, "regression": regression, "metrics": metrics}


def evaluatePod(model: dict, X: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, dict]:
    APred = predictCoefficients(P, model["regression"])
    XPred = reconstruct(model["pod"], APred)
    metrics = computeMetrics(X, XPred)
    return XPred, metrics


def evaluateProjection(model: dict, X: np.ndarray) -> tuple[np.ndarray, dict]:
    pod = model["pod"]
    AProj = pod["PHI"].T @ (X - pod["mean"])
    XProj = reconstruct(pod, AProj)
    metrics = computeMetrics(X, XProj)
    return XProj, metrics


def splitTrainTest(metadata: list[dict], nSnapshots: int, testRatio: float, splitMode: str, seed: int, testCaseNames: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if testCaseNames:
        wanted = set(testCaseNames)
        found = {case["caseName"] for case in metadata}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Unknown test cases: {missing}; available={sorted(found)}")
        trainIdx = []
        testIdx = []
        for case in metadata:
            target = testIdx if case["caseName"] in wanted else trainIdx
            target.extend(range(case["snapshotStart"], case["snapshotEnd"]))
        return np.asarray(trainIdx, dtype=np.int64), np.asarray(testIdx, dtype=np.int64)
    if splitMode == "random":
        idx = np.arange(nSnapshots)
        rng.shuffle(idx)
        nTest = max(1, int(round(nSnapshots * testRatio)))
        testIdx = np.sort(idx[:nTest])
        trainIdx = np.sort(idx[nTest:])
        return trainIdx, testIdx
    if splitMode == "case":
        caseOrder = np.arange(len(metadata))
        rng.shuffle(caseOrder)
        nTestCases = max(1, int(round(len(metadata) * testRatio)))
        testCases = set(caseOrder[:nTestCases].tolist())
        trainIdx = []
        testIdx = []
        for caseId, case in enumerate(metadata):
            target = testIdx if caseId in testCases else trainIdx
            target.extend(range(case["snapshotStart"], case["snapshotEnd"]))
        return np.asarray(trainIdx, dtype=np.int64), np.asarray(testIdx, dtype=np.int64)
    trainIdx = []
    testIdx = []
    for case in metadata:
        start = case["snapshotStart"]
        end = case["snapshotEnd"]
        nCase = end - start
        if nCase <= 1:
            trainIdx.extend(range(start, end))
            continue
        nTest = max(1, int(round(nCase * testRatio)))
        split = max(start + 1, end - nTest)
        trainIdx.extend(range(start, split))
        testIdx.extend(range(split, end))
    return np.asarray(trainIdx, dtype=np.int64), np.asarray(testIdx, dtype=np.int64)


def getSplitCaseNames(metadata: list[dict], idx: np.ndarray) -> list[str]:
    names = []
    idxSet = set(idx.tolist())
    for case in metadata:
        caseIdx = set(range(case["snapshotStart"], case["snapshotEnd"]))
        if idxSet & caseIdx:
            names.append(case["caseName"])
    return names


def loadSolidCoords(casPath: Path, nSolid: int) -> np.ndarray | None:
    try:
        import h5py
        with h5py.File(casPath, "r") as file:
            nodeCoords = np.asarray(file["meshes/1/nodes/coords/1"][()])
            faceNN = np.asarray(file["meshes/1/faces/nodes/1/nnodes"][()])
            faceNodes = np.asarray(file["meshes/1/faces/nodes/1/nodes"][()])
            faceC0 = np.asarray(file["meshes/1/faces/c0/1"][()])
        offsets = np.cumsum(np.concatenate([[0], faceNN]))
        nCells = int(faceC0.max()) + 1
        cellNodeSets = [set() for _ in range(nCells)]
        for fid in range(len(faceNN)):
            owner = int(faceC0[fid])
            if owner < 0 or owner >= nCells:
                continue
            for nid in faceNodes[offsets[fid]:offsets[fid + 1]]:
                cellNodeSets[owner].add(int(nid) - 1)
        cellCenters = np.zeros((nCells, 3), dtype=np.float64)
        for cid, nodeSet in enumerate(cellNodeSets):
            if len(nodeSet) == 0:
                continue
            cellCenters[cid] = nodeCoords[np.asarray(list(nodeSet), dtype=np.int64)].mean(axis=0)
        return cellCenters[1:nSolid + 1]
    except Exception as exc:
        print(f"WARNING: cannot load solid coordinates for visualization: {exc}")
        return None


def selectPointIdx(coords: np.ndarray | None, nSolid: int) -> np.ndarray:
    if coords is None or len(coords) != nSolid:
        return np.asarray([0, nSolid // 2, nSolid - 1], dtype=np.int64)
    x = coords[:, 0]
    center = coords.mean(axis=0)
    return np.asarray([
        int(np.argmin(x)),
        int(np.argmin(np.linalg.norm(coords - center, axis=1))),
        int(np.argmax(x)),
    ], dtype=np.int64)


def plotCloudComparison(coords: np.ndarray | None, actual: np.ndarray, pred: np.ndarray, outPath: Path, maxPoints: int, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    nSolid = actual.shape[0]
    plotIdx = np.arange(nSolid)
    if nSolid > maxPoints:
        plotIdx = np.sort(rng.choice(nSolid, size=maxPoints, replace=False))
    actualPlot = actual[plotIdx]
    predPlot = pred[plotIdx]
    errPlot = predPlot - actualPlot
    if coords is not None and len(coords) == nSolid:
        xyz = coords[plotIdx]
        fig = plt.figure(figsize=(15, 5))
        panels = [(actualPlot, "Actual T (K)", "viridis"), (predPlot, "POD Pred T (K)", "viridis"), (errPlot, "Error (K)", "coolwarm")]
        for i, (vals, title, cmap) in enumerate(panels, start=1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            scat = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=vals, s=2, cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            fig.colorbar(scat, ax=ax, shrink=0.7)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        panels = [(actualPlot, "Actual T (K)", "viridis"), (predPlot, "POD Pred T (K)", "viridis"), (errPlot, "Error (K)", "coolwarm")]
        for ax, (vals, title, cmap) in zip(axes, panels):
            scat = ax.scatter(plotIdx, vals, c=vals, s=2, cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel("solid cell index")
            fig.colorbar(scat, ax=ax)
    fig.tight_layout()
    outPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outPath, dpi=160)
    plt.close(fig)


def plotThreePointComparison(P: np.ndarray, X: np.ndarray, XPred: np.ndarray, testIdx: np.ndarray, pointIdx: np.ndarray, metadata: list[dict], outPath: Path, timeRange: tuple[float, float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, pid in zip(axes, pointIdx):
        for case in metadata:
            mask = (testIdx >= case["snapshotStart"]) & (testIdx < case["snapshotEnd"])
            if not np.any(mask):
                continue
            localTestPos = np.where(mask)[0]
            globalIdx = testIdx[localTestPos]
            order = np.argsort(P[globalIdx, 0])
            globalIdx = globalIdx[order]
            localTestPos = localTestPos[order]
            xAxis = P[globalIdx, 0]
            ax.plot(xAxis, X[pid, globalIdx], "o-", label=f"{case['caseName']} actual", linewidth=1)
            ax.plot(xAxis, XPred[pid, localTestPos], "s--", label=f"{case['caseName']} pod", linewidth=1)
        ax.set_ylabel(f"cell {pid}\nT(K)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(timeRange)
    axes[-1].set_xlabel("time")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    outPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outPath, dpi=160)
    plt.close(fig)


def saveModel(model: dict, outputPath: Path, metadata: list[dict], P: np.ndarray, summaryExtra: dict | None = None) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    pod = model["pod"]
    regression = model["regression"]
    saved = {
        "mean": pod["mean"],
        "PHI": pod["PHI"],
        "singularValues": pod["S"],
        "energyRatio": pod["energyRatio"],
        "paramMin": regression["paramScale"]["min"],
        "paramRange": regression["paramScale"]["range"],
        "params": P,
        "regressionType": np.array([regression["type"]]),
    }
    if regression["type"] == "poly":
        saved["W"] = regression["W"]
        saved["degree"] = np.array([regression["degree"]], dtype=np.int32)
        saved["ridge"] = np.array([regression["ridge"]], dtype=np.float64)
    if regression["type"] == "rbf":
        saved["ridge"] = np.array([regression["ridge"]], dtype=np.float64)
        saved["rbfEpsilon"] = np.array([regression["epsilon"]], dtype=np.float64)
        saved["rbfNeighbors"] = np.array([regression["neighbors"]], dtype=np.int32)
        saved["PTrainNorm"] = regression["PTrainNorm"]
        saved["ATrain"] = regression["ATrain"]
    if regression["type"] == "idw":
        saved["idwNeighbors"] = np.array([regression["neighbors"]], dtype=np.int32)
        saved["idwPower"] = np.array([regression["power"]], dtype=np.float64)
        saved["idwEps"] = np.array([regression["eps"]], dtype=np.float64)
        saved["PTrainNorm"] = regression["PTrainNorm"]
        saved["ATrain"] = regression["ATrain"]
    if regression["type"] == "mlp":
        saved["coeffMean"] = regression["coeffMean"]
        saved["coeffStd"] = regression["coeffStd"]
        saved["mlpHidden"] = np.array([regression["hidden"]], dtype=np.int32)
        saved["mlpLayers"] = np.array([regression["layers"]], dtype=np.int32)
        saved["mlpBestLoss"] = np.array([regression["bestLoss"]], dtype=np.float64)
        for key, value in regression["model"].state_dict().items():
            saved[f"mlp_{key}"] = value.detach().cpu().numpy()
    np.savez_compressed(outputPath, **saved)
    summaryPath = outputPath.with_suffix(".json")
    summary = {
        "nSolid": int(pod["mean"].shape[0]),
        "nSnapshots": int(P.shape[0]),
        "nModes": int(pod["nModes"]),
        "energy": float(pod["energyRatio"][pod["nModes"] - 1]),
        "regressionType": regression["type"],
        "metrics": model["metrics"],
        "cases": metadata,
    }
    if summaryExtra is not None:
        summary.update(summaryExtra)
    summaryPath.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def runSelfTest() -> None:
    rng = np.random.default_rng(42)
    nSolid = 240
    nSnapshots = 36
    x = np.linspace(0.0, 1.0, nSolid)
    P = np.zeros((nSnapshots, 3), dtype=np.float64)
    X = np.zeros((nSolid, nSnapshots), dtype=np.float64)
    for i in range(nSnapshots):
        t = i / (nSnapshots - 1)
        tp = 433.15 + 10.0 * ((i // 12) - 1)
        th = 473.15 + 15.0 * ((i // 6) % 3 - 1)
        P[i] = [t * 7.4, tp, th]
        X[:, i] = 300.0 + 25.0 * t + 0.06 * (tp - 433.15) * np.sin(np.pi * x) + 0.04 * (th - 473.15) * np.cos(2.0 * np.pi * x)
        X[:, i] += rng.normal(0.0, 0.01, size=nSolid)
    model = trainPod(X, P, energy=0.999, maxModes=None, degree=2, ridge=1e-8, regressionType="rbf", rbfEpsilon=1.0, rbfRidge=1e-5, rbfNeighbors=64, idwNeighbors=64, idwPower=2.0, idwEps=1e-12, mlpHidden=64, mlpLayers=2, mlpEpochs=500, mlpLearningRate=1e-3, mlpWeightDecay=1e-6, mlpValRatio=0.1, mlpPatience=100)
    print(json.dumps({
        "selfTest": "ok",
        "nModes": model["pod"]["nModes"],
        "energy": float(model["pod"]["energyRatio"][model["pod"]["nModes"] - 1]),
        "metrics": model["metrics"],
    }, indent=2))


def main() -> None:
    config = PodConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataRoot", type=Path, default=config.dataRoot)
    parser.add_argument("--caseGlob", default=config.caseGlob)
    parser.add_argument("--datGlob", default=config.datGlob)
    parser.add_argument("--output", type=Path, default=config.outputPath)
    parser.add_argument("--energy", type=float, default=config.energy)
    parser.add_argument("--maxModes", type=int, default=config.maxModes)
    parser.add_argument("--degree", type=int, default=config.degree)
    parser.add_argument("--ridge", type=float, default=config.ridge)
    parser.add_argument("--regressionType", choices=["poly", "rbf", "idw", "mlp"], default=config.regressionType)
    parser.add_argument("--rbfEpsilon", type=float, default=config.rbfEpsilon)
    parser.add_argument("--rbfRidge", type=float, default=config.rbfRidge)
    parser.add_argument("--rbfNeighbors", type=int, default=config.rbfNeighbors)
    parser.add_argument("--idwNeighbors", type=int, default=config.idwNeighbors)
    parser.add_argument("--idwPower", type=float, default=config.idwPower)
    parser.add_argument("--idwEps", type=float, default=config.idwEps)
    parser.add_argument("--mlpHidden", type=int, default=getattr(config, "mlpHidden", 128))
    parser.add_argument("--mlpLayers", type=int, default=getattr(config, "mlpLayers", 3))
    parser.add_argument("--mlpEpochs", type=int, default=getattr(config, "mlpEpochs", 3000))
    parser.add_argument("--mlpLearningRate", type=float, default=getattr(config, "mlpLearningRate", 1e-3))
    parser.add_argument("--mlpWeightDecay", type=float, default=getattr(config, "mlpWeightDecay", 1e-6))
    parser.add_argument("--mlpValRatio", type=float, default=getattr(config, "mlpValRatio", 0.1))
    parser.add_argument("--mlpPatience", type=int, default=getattr(config, "mlpPatience", 300))
    parser.add_argument("--testRatio", type=float, default=config.testRatio)
    parser.add_argument("--splitMode", choices=["temporal", "random", "case"], default=config.splitMode)
    parser.add_argument("--testCases", default=config.testCases)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--casPath", type=Path, default=config.casPath)
    parser.add_argument("--figDir", type=Path, default=config.figDir)
    parser.add_argument("--maxPlotPoints", type=int, default=config.maxPlotPoints)
    parser.add_argument("--selfTest", action="store_true")
    args = parser.parse_args()
    if args.selfTest:
        runSelfTest()
        return
    maxModes = args.maxModes if args.maxModes > 0 else None
    X, P, metadata = loadSolidSnapshots(args.dataRoot, args.caseGlob, args.datGlob)
    testCaseNames = [name.strip() for name in args.testCases.split(",") if name.strip()]
    trainIdx, testIdx = splitTrainTest(metadata, X.shape[1], args.testRatio, args.splitMode, args.seed, testCaseNames)
    if len(trainIdx) == 0 or len(testIdx) == 0:
        raise ValueError(f"Invalid split: train={len(trainIdx)} test={len(testIdx)}")
    trainCaseNames = getSplitCaseNames(metadata, trainIdx)
    splitTestCaseNames = getSplitCaseNames(metadata, testIdx)
    model = trainPod(
        X[:, trainIdx],
        P[trainIdx],
        args.energy,
        maxModes,
        args.degree,
        args.ridge,
        args.regressionType,
        args.rbfEpsilon,
        args.rbfRidge,
        args.rbfNeighbors,
        args.idwNeighbors,
        args.idwPower,
        args.idwEps,
        args.mlpHidden,
        args.mlpLayers,
        args.mlpEpochs,
        args.mlpLearningRate,
        args.mlpWeightDecay,
        args.mlpValRatio,
        args.mlpPatience,
    )
    _, trainMetrics = evaluatePod(model, X[:, trainIdx], P[trainIdx])
    testPred, testMetrics = evaluatePod(model, X[:, testIdx], P[testIdx])
    _, trainProjectionMetrics = evaluateProjection(model, X[:, trainIdx])
    _, testProjectionMetrics = evaluateProjection(model, X[:, testIdx])
    coords = loadSolidCoords(args.casPath, X.shape[0])
    pointIdx = selectPointIdx(coords, X.shape[0])
    cloudSnapshotLocal = 0
    cloudSnapshotGlobal = int(testIdx[cloudSnapshotLocal])
    plotCloudComparison(
        coords,
        X[:, cloudSnapshotGlobal],
        testPred[:, cloudSnapshotLocal],
        args.figDir / f"cloudCompare_snapshot_{cloudSnapshotGlobal:04d}.png",
        args.maxPlotPoints,
        args.seed,
    )
    plotThreePointComparison(
        P,
        X,
        testPred,
        testIdx,
        pointIdx,
        metadata,
        args.figDir / "threePointCompare.png",
        (float(P[:, 0].min()), float(P[:, 0].max())),
    )
    summaryExtra = {
        "splitMode": args.splitMode,
        "testRatio": args.testRatio,
        "regressionType": args.regressionType,
        "rbfEpsilon": args.rbfEpsilon,
        "rbfRidge": args.rbfRidge,
        "rbfNeighbors": args.rbfNeighbors,
        "idwNeighbors": args.idwNeighbors,
        "idwPower": args.idwPower,
        "idwEps": args.idwEps,
        "trainCases": trainCaseNames,
        "testCases": splitTestCaseNames,
        "trainIdx": trainIdx.tolist(),
        "testIdx": testIdx.tolist(),
        "trainMetrics": trainMetrics,
        "testMetrics": testMetrics,
        "trainProjectionMetrics": trainProjectionMetrics,
        "testProjectionMetrics": testProjectionMetrics,
        "pointIdx": pointIdx.tolist(),
        "figures": {
            "cloudCompare": str(args.figDir / f"cloudCompare_snapshot_{cloudSnapshotGlobal:04d}.png"),
            "threePointCompare": str(args.figDir / "threePointCompare.png"),
        },
    }
    saveModel(model, args.output, metadata, P, summaryExtra)
    print(json.dumps({
        "output": str(args.output),
        "figDir": str(args.figDir),
        "nSolid": int(X.shape[0]),
        "nSnapshots": int(X.shape[1]),
        "nTrain": int(len(trainIdx)),
        "nTest": int(len(testIdx)),
        "splitMode": args.splitMode,
        "regressionType": args.regressionType,
        "rbfEpsilon": args.rbfEpsilon,
        "rbfRidge": args.rbfRidge,
        "rbfNeighbors": args.rbfNeighbors,
        "idwNeighbors": args.idwNeighbors,
        "idwPower": args.idwPower,
        "idwEps": args.idwEps,
        "trainCases": trainCaseNames,
        "testCases": splitTestCaseNames,
        "nModes": int(model["pod"]["nModes"]),
        "energy": float(model["pod"]["energyRatio"][model["pod"]["nModes"] - 1]),
        "trainMetrics": trainMetrics,
        "testMetrics": testMetrics,
        "trainProjectionMetrics": trainProjectionMetrics,
        "testProjectionMetrics": testProjectionMetrics,
        "pointIdx": pointIdx.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
