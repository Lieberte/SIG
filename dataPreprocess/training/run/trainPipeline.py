import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def addProjectRootToPath() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


addProjectRootToPath()

from training.endToEnd.model import bcToSurfaceModel
from training.gnn.graphDataset import meshGraphDataset
from training.gnn.model import temporalMeshGnn
from training.pinn.model import pinnLossComposer, pinnModel
from training.podRom.model import latentDynamicsModel
from training.shared.graphBuilder import (
    buildCellAdjacencyFast,
    buildCellAdjacencyTyped,
    buildEdgeAttr,
    extractNodeFeatures,
)
from training.shared.normalizer import standardScaler
from training.shared.podBasis import podBasis
from training.shared.sequenceDataset import timeSeriesDataset
from training.shared.surfaceExtractor import (
    cellIdsForZones,
    filterZones,
    loadFaceOwnerCells,
    loadSchema,
    loadTopologyMeta,
    sliceFieldFromMatrix,
    zoneMeanFromMatrix,
)
from training.shared.trainer import romTrainer


def loadConfig(configPath: str | Path) -> dict:
    return json.loads(Path(configPath).read_text(encoding="utf-8"))


def splitTrainVal(total: int, valSplit: float) -> tuple[int, int]:
    nVal = int(total * valSplit)
    return total - nVal, nVal


def extractBcAndTarget(
    matrix: np.ndarray,
    schema: dict,
    topologyMeta: dict,
    casePath: Path,
    config: dict,
) -> dict[str, np.ndarray]:
    faceOwnerCells = loadFaceOwnerCells(casePath)
    nCells = schema["nCells"]
    inletZones = filterZones(topologyMeta, role=config.get("inletRole", "boundary"), zoneType=config.get("inletType", "inlet"))
    wallZones = filterZones(topologyMeta, role=config.get("wallRole", "boundary"), zoneType=config.get("wallType", "wall"))
    inletCellIds = cellIdsForZones(faceOwnerCells, inletZones, maxCellId=nCells)
    wallCellIds = cellIdsForZones(faceOwnerCells, wallZones, maxCellId=nCells)
    bcInput = zoneMeanFromMatrix(matrix, schema, inletCellIds, config["bcFieldNames"])
    targetField = config.get("targetFieldName", "temperature")
    surfaceTarget = sliceFieldFromMatrix(matrix, schema, targetField, wallCellIds).T
    return {
        "bcInput": bcInput,
        "surfaceTarget": surfaceTarget,
        "inletCellIds": inletCellIds,
        "wallCellIds": wallCellIds,
    }


def _saveHistory(history: dict, outputDir: Path) -> None:
    np.save(outputDir / "trainLoss.npy", np.array(history["trainLoss"]))
    if history["valLoss"]:
        np.save(outputDir / "valLoss.npy", np.array(history["valLoss"]))


# ── Phase A ─────────────────────────────────────────────────

def runEndToEnd(config: dict) -> dict:
    matrix = np.load(config["snapshotMatrixPath"])
    schema = loadSchema(Path(config["schemaPath"]))
    topologyMeta = loadTopologyMeta(Path(config["topologyMetaPath"]))
    casePath = Path(config["casePath"])
    pair = extractBcAndTarget(matrix, schema, topologyMeta, casePath, config)
    bcInput, surfaceTarget = pair["bcInput"], pair["surfaceTarget"]
    bcScaler, targetScaler = standardScaler(), standardScaler()
    if config.get("normalize", True):
        bcInput = bcScaler.fitTransform(bcInput)
        surfaceTarget = targetScaler.fitTransform(surfaceTarget)
    seqLen = config.get("seqLen", 10)
    dataset = timeSeriesDataset(bcInput, surfaceTarget, seqLen=seqLen)
    if len(dataset) == 0:
        raise ValueError(f"not enough snapshots ({matrix.shape[1]}) for seqLen={seqLen}")
    nTrain, nVal = splitTrainVal(len(dataset), config.get("valSplit", 0.2))
    trainDs, valDs = torch.utils.data.random_split(dataset, [nTrain, nVal])
    batchSize = config.get("batchSize", 32)
    trainLoader = DataLoader(trainDs, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valDs, batch_size=batchSize) if nVal > 0 else None
    model = bcToSurfaceModel(
        nInputFeatures=bcInput.shape[1],
        nOutputFeatures=surfaceTarget.shape[1] if surfaceTarget.ndim > 1 else 1,
        hiddenSize=config.get("hiddenSize", 128),
        nLayers=config.get("nLayers", 2),
        dropout=config.get("dropout", 0.1),
    )
    device = config.get("device", "cpu")
    trainer = romTrainer(model, torch.optim.Adam(model.parameters(), lr=config.get("learningRate", 1e-3)), torch.nn.MSELoss(), device=device)
    history = trainer.fit(trainLoader, valLoader, nEpochs=config.get("nEpochs", 200), patience=config.get("patience", 20))
    outputDir = Path(config["outputDir"])
    outputDir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outputDir / "model.pt")
    _saveHistory(history, outputDir)
    if config.get("normalize", True):
        np.savez(outputDir / "scalers.npz", bcMean=bcScaler.mean, bcStd=bcScaler.std, targetMean=targetScaler.mean, targetStd=targetScaler.std)
    report = {
        "mode": "endToEnd",
        "nWallCells": int(pair["wallCellIds"].size),
        "nInletCells": int(pair["inletCellIds"].size),
        "nSnapshots": int(matrix.shape[1]),
        "nTrainSamples": nTrain,
        "nValSamples": nVal,
        "finalTrainLoss": float(history["trainLoss"][-1]) if history["trainLoss"] else None,
        "finalValLoss": float(history["valLoss"][-1]) if history["valLoss"] else None,
    }
    (outputDir / "trainReport.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ── Phase B ─────────────────────────────────────────────────

def runPodRom(config: dict) -> dict:
    """Run POD-ROM training with proper temporal split and POD fit on train data only."""
    from training.shared.dataSplit import temporal_split
    from training.podRom.preprocess import PodRomPreprocessor

    matrix = np.load(config["snapshotMatrixPath"])
    schema = loadSchema(Path(config["schemaPath"]))
    topologyMeta = loadTopologyMeta(Path(config["topologyMetaPath"]))
    casePath = Path(config["casePath"])

    nModes = config.get("nModes", 20)
    valRatio = config.get("valSplit", 0.2)
    testRatio = config.get("testRatio", 0.0)
    normalize = config.get("normalize", True)

    # Temporal split (no leakage)
    nSnapshots = matrix.shape[1]
    trainIdx, valIdx, testIdx = temporal_split(nSnapshots, valRatio, testRatio)

    # POD fit on TRAIN data only
    trainMatrix = matrix[:, trainIdx]
    pod = podBasis(nModes)
    pod.fit(trainMatrix)

    # Encode all snapshots using train-fitted POD
    coefficients = pod.encode(matrix)
    latentT = coefficients.T  # (nSnapshots, nModes)

    # Extract BC
    pair = extractBcAndTarget(matrix, schema, topologyMeta, casePath, config)
    bcInput = pair["bcInput"]

    # Normalizers fit on train data only
    bcScaler, latentScaler = standardScaler(), standardScaler()
    if normalize:
        bcInput_train = bcInput[trainIdx]
        bcScaler.fit(bcInput_train)
        bcInput = bcScaler.transform(bcInput)

        latentT_train = latentT[trainIdx]
        latentScaler.fit(latentT_train)
        latentT = latentScaler.transform(latentT)

    # Build dataset with temporal split
    combinedInput = np.concatenate([latentT, bcInput], axis=1)
    seqLen = config.get("seqLen", 10)
    dataset = timeSeriesDataset(combinedInput, latentT, seqLen=seqLen)

    if len(dataset) == 0:
        raise ValueError(f"not enough snapshots ({nSnapshots}) for seqLen={seqLen}")

    # Apply temporal split to dataset indices
    nSamples = len(dataset)
    trainDsIdx, valDsIdx, testDsIdx = temporal_split(nSamples, valRatio, testRatio)

    trainDs = torch.utils.data.Subset(dataset, trainDsIdx)
    valDs = torch.utils.data.Subset(dataset, valDsIdx) if len(valDsIdx) > 0 else None

    batchSize = config.get("batchSize", 32)
    trainLoader = DataLoader(trainDs, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valDs, batch_size=batchSize) if valDs is not None else None

    nLatent = coefficients.shape[0]
    model = latentDynamicsModel(
        nLatent=nLatent,
        nBcFeatures=bcInput.shape[1],
        hiddenSize=config.get("hiddenSize", 128),
        nLayers=config.get("nLayers", 2),
        dropout=config.get("dropout", 0.1),
        activation=config.get("activation", "tanh"),
    )

    device = config.get("device", "cpu")
    trainer = romTrainer(
        model,
        torch.optim.Adam(model.parameters(), lr=config.get("learningRate", 1e-3)),
        torch.nn.MSELoss(),
        device=device,
    )
    history = trainer.fit(
        trainLoader, valLoader,
        nEpochs=config.get("nEpochs", 200),
        patience=config.get("patience", 20),
    )

    outputDir = Path(config["outputDir"])
    outputDir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), outputDir / "model.pt")
    pod.save(outputDir)
    _saveHistory(history, outputDir)

    if normalize:
        np.savez(
            outputDir / "scalers.npz",
            bcMean=bcScaler.mean,
            bcStd=bcScaler.std,
            latentMean=latentScaler.mean,
            latentStd=latentScaler.std,
        )

    # Compute reconstruction error on full data
    reconError = pod.reconstructionError(matrix)

    report = {
        "mode": "podRom",
        "nModes": nLatent,
        "truncatedEnergy": pod.truncatedEnergy(),
        "reconstructionError": reconError,
        "nSnapshots": nSnapshots,
        "nTrainSamples": len(trainDsIdx),
        "nValSamples": len(valDsIdx),
        "nTestSamples": len(testDsIdx),
        "splitStrategy": "temporal",
        "podFitOn": "trainOnly",
        "finalTrainLoss": float(history["trainLoss"][-1]) if history["trainLoss"] else None,
        "finalValLoss": float(history["valLoss"][-1]) if history["valLoss"] else None,
        "podEnergyReport": pod.energyReport(),
    }
    (outputDir / "trainReport.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return report


# ── PINN ────────────────────────────────────────────────────

def runPinn(config: dict) -> dict:
    """Run PINN training with proper data preprocessing."""
    from training.shared.surfaceExtractor import loadFaceOwnerCells, filterZones, cellIdsForZones
    from training.shared.surfaceExtractor import loadSchema, loadTopologyMeta

    matrix = np.load(config["snapshotMatrixPath"])
    schema = loadSchema(Path(config["schemaPath"]))
    casePath = Path(config["casePath"])
    topologyMetaPath = config.get("topologyMetaPath")
    topologyMeta = loadTopologyMeta(Path(topologyMetaPath)) if topologyMetaPath else None
    fieldNames = config.get("fieldNames", [])
    nSpatialDim = config.get("nSpatialDim", 3)
    nBcParams = config.get("nBcParams", 0)

    nCells = schema["nCells"]
    nFields = len(fieldNames) if fieldNames else schema.get("nFields", 6)
    cellCenters = None  # PINN preprocessor generates its own coords

    # Prepare PINN dataset
    dataset = preparePinnData(
        snapshotMatrix=matrix,
        schema=schema,
        topologyMeta=topologyMeta,
        casePath=casePath,
        fieldNames=fieldNames,
        nCollocationPoints=config.get("nCollocationPoints", 10000),
        nBoundaryPoints=config.get("nBoundaryPoints", 1000),
        nInitialPoints=config.get("nInitialPoints", 1000),
        nDataPoints=config.get("nDataPoints", 5000),
        timeRange=tuple(config.get("timeRange", [0.0, 1.0])),
    )

    device = config.get("device", "cpu")
    model = pinnModel(
        nSpatialDim=nSpatialDim,
        nBcParams=nBcParams,
        nOutputFields=nFields,
        hiddenSize=config.get("hiddenSize", 128),
        nBlocks=config.get("nBlocks", 6),
    )
    model.fieldNames = fieldNames

    physicsParams = config.get("physicsParams", {})
    composer = pinnLossComposer(
        model,
        physicsParams,
        lambdaData=config.get("lambdaData", 1.0),
        lambdaPhysics=config.get("lambdaPhysics", 0.1),
    )
    enabledPdes = config.get("enabledPdes")
    if enabledPdes is not None:
        composer.enabledPdes = set(enabledPdes)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learningRate", 1e-3))
    nEpochs = config.get("nEpochs", 5000)
    outputDir = Path(config["outputDir"])
    outputDir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    batchSize = config.get("batchSize", 4096)

    def _colloc_loader():
        coords = torch.from_numpy(dataset.collocationCoords).float().to(device)
        times = torch.from_numpy(dataset.collocationTimes).float().to(device)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(coords, times),
            batch_size=batchSize, shuffle=True,
        )

    def _boundary_loader():
        if len(dataset.boundaryCoords) == 0:
            return None
        coords = torch.from_numpy(dataset.boundaryCoords).float().to(device)
        times = torch.from_numpy(dataset.boundaryTimes).float().to(device)
        values = torch.from_numpy(dataset.boundaryValues).float().to(device)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(coords, times, values),
            batch_size=batchSize, shuffle=True,
        )

    def _initial_loader():
        coords = torch.from_numpy(dataset.initialCoords).float().to(device)
        values = torch.from_numpy(dataset.initialValues).float().to(device)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(coords, values),
            batch_size=batchSize, shuffle=True,
        )

    def _data_loader():
        coords = torch.from_numpy(dataset.dataCoords).float().to(device)
        times = torch.from_numpy(dataset.dataTimes).float().to(device)
        values = torch.from_numpy(dataset.dataValues).float().to(device)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(coords, times, values),
            batch_size=batchSize, shuffle=True,
        )

    nTrain = len(dataset.collocationCoords)
    bestLoss = float("inf")
    patience = config.get("patience", 200)
    noImproveCount = 0
    history = {"trainLoss": [], "valLoss": []}

    for epoch in range(nEpochs):
        model.train()
        totalLoss = 0.0
        nBatches = 0

        # Collocation loss
        for coords, times in _colloc_loader():
            pred = model(coords, times, None)
            loss = composer.dataLoss(pred, torch.zeros_like(pred)).item()
            physicsLosses = composer.physicsLoss(
                coords, times, None,
            )
            pLoss = sum(physicsLosses.values()) if physicsLosses else torch.tensor(0.0)
            totalLoss += (loss + config.get("lambdaPhysics", 0.1) * pLoss).item()
            nBatches += 1

        # Boundary loss (if available)
        bLoader = _boundary_loader()
        if bLoader is not None:
            for coords, times, values in bLoader():
                pred = model(coords, times, values)
                loss = composer.dataLoss(pred, values).item()
                totalLoss += loss
                nBatches += 1

        # Initial condition loss
        iLoader = _initial_loader()
        for coords, values in iLoader():
            pred = model(coords, torch.zeros(coords.shape[0], 1, device=device), None)
            loss = composer.dataLoss(pred, values).item()
            totalLoss += loss
            nBatches += 1

        # Data loss (from CFD snapshots)
        dLoader = _data_loader()
        for coords, times, values in dLoader():
            pred = model(coords, times, None)
            loss = composer.dataLoss(pred, values).item()
            totalLoss += loss
            nBatches += 1

        avgLoss = totalLoss / max(nBatches, 1)
        history["trainLoss"].append(avgLoss)

        # Validation on boundary data
        valLoss = None
        if bLoader is not None:
            model.eval()
            with torch.no_grad():
                valTotal = 0.0
                valBatches = 0
                for coords, times, values in bLoader():
                    pred = model(coords, times, values)
                    valTotal += composer.dataLoss(pred, values).item()
                    valBatches += 1
                valLoss = valTotal / max(valBatches, 1)
            model.train()
            history["valLoss"].append(valLoss)

        if avgLoss < bestLoss:
            bestLoss = avgLoss
            noImproveCount = 0
            torch.save(model.state_dict(), outputDir / "model_best.pt")
        else:
            noImproveCount += 1

        if epoch % 100 == 0:
            msg = f"Epoch {epoch}/{nEpochs}  loss={avgLoss:.6f}"
            if valLoss is not None:
                msg += f"  valLoss={valLoss:.6f}"
            print(msg)

        if patience > 0 and noImproveCount >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        optimizer.zero_grad()
        # Recompute backward pass on final batch
        model.train()

    torch.save(model.state_dict(), outputDir / "model.pt")
    np.save(outputDir / "trainLoss.npy", np.array(history["trainLoss"]))
    if history["valLoss"]:
        np.save(outputDir / "valLoss.npy", np.array(history["valLoss"]))

    report = {
        "mode": "pinn",
        "nEpochsRan": len(history["trainLoss"]),
        "finalLoss": history["trainLoss"][-1] if history["trainLoss"] else None,
        "bestLoss": bestLoss,
        "enabledPdes": list(composer.enabledPdes),
        "nCollocationPoints": len(dataset.collocationCoords),
        "nBoundaryPoints": len(dataset.boundaryCoords),
        "nInitialPoints": len(dataset.initialCoords),
        "nDataPoints": len(dataset.dataCoords),
    }
    (outputDir / "trainReport.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return report


# ── GNN ─────────────────────────────────────────────────────

def runGnn(config: dict) -> dict:
    """Run GNN training with proper temporal split."""
    from training.shared.dataSplit import temporal_split

    matrix = np.load(config["snapshotMatrixPath"])
    schema = loadSchema(Path(config["schemaPath"]))
    casePath = Path(config["casePath"])
    topologyMetaPath = config.get("topologyMetaPath")
    topologyMeta = loadTopologyMeta(Path(topologyMetaPath)) if topologyMetaPath else None
    fieldNames = config.get("fieldNames")

    nodeFeatures = extractNodeFeatures(matrix, schema, fieldNames)
    graph = buildCellAdjacencyTyped(casePath, topologyMeta)
    edgeIndex = graph["edgeIndex"]
    useEdgeAttr = config.get("useEdgeAttr", True)
    edgeAttrNp = buildEdgeAttr(graph["edgeType"]) if useEdgeAttr else None

    seqLen = config.get("seqLen", 10)
    dataset = meshGraphDataset(nodeFeatures, edgeIndex, seqLen=seqLen, edgeAttr=edgeAttrNp)

    if len(dataset) == 0:
        raise ValueError(f"not enough snapshots ({nodeFeatures.shape[0]}) for seqLen={seqLen}")

    # Temporal split (no leakage)
    nSamples = len(dataset)
    trainIdx, valIdx, testIdx = temporal_split(
        nSamples,
        config.get("valSplit", 0.2),
        config.get("testRatio", 0.0),
    )

    trainDs = torch.utils.data.Subset(dataset, trainIdx)
    valDs = torch.utils.data.Subset(dataset, valIdx) if len(valIdx) > 0 else None

    batchSize = config.get("batchSize", 4)
    trainLoader = DataLoader(trainDs, batch_size=batchSize, shuffle=True)
    valLoader = DataLoader(valDs, batch_size=batchSize) if valDs is not None else None

    nNodeFeatures = nodeFeatures.shape[2]
    convType = config.get("convType", "gat")
    edgeDim = edgeAttrNp.shape[1] if edgeAttrNp is not None else None

    model = temporalMeshGnn(
        nNodeFeatures=nNodeFeatures,
        nOutputFeatures=nNodeFeatures,
        hiddenSize=config.get("hiddenSize", 64),
        nConvLayers=config.get("nConvLayers", 3),
        convType=convType,
        nHeads=config.get("nHeads", 4),
        nGruLayers=config.get("nGruLayers", 1),
        dropout=config.get("dropout", 0.0),
        edgeDim=edgeDim if convType == "gat" else None,
    )

    device = config.get("device", "cpu")
    edgeIndexTensor = torch.from_numpy(edgeIndex).to(device)
    edgeAttrTensor = torch.from_numpy(edgeAttrNp).to(device) if edgeAttrNp is not None else None

    class _gnnWrapper(torch.nn.Module):
        def __init__(self, inner, ei, ea):
            super().__init__()
            self.inner = inner
            self.register_buffer("ei", ei)
            if ea is not None:
                self.register_buffer("ea", ea)
            else:
                self.ea = None
        def forward(self, x):
            return self.inner(x, self.ei, self.ea)

    wrapper = _gnnWrapper(model, edgeIndexTensor, edgeAttrTensor)
    trainer = romTrainer(
        wrapper,
        torch.optim.Adam(model.parameters(), lr=config.get("learningRate", 1e-3)),
        torch.nn.MSELoss(),
        device=device,
    )
    history = trainer.fit(
        trainLoader, valLoader,
        nEpochs=config.get("nEpochs", 200),
        patience=config.get("patience", 20),
    )

    outputDir = Path(config["outputDir"])
    outputDir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), outputDir / "model.pt")
    np.save(outputDir / "edgeIndex.npy", edgeIndex)
    if edgeAttrNp is not None:
        np.save(outputDir / "edgeAttr.npy", edgeAttrNp)
    _saveHistory(history, outputDir)

    report = {
        "mode": "gnn",
        "convType": convType,
        "useEdgeAttr": useEdgeAttr,
        "nCells": graph["nCells"],
        "nEdges": graph["nEdges"],
        "nNodeFeatures": nNodeFeatures,
        "nSnapshots": int(nodeFeatures.shape[0]),
        "nTrainSamples": len(trainIdx),
        "nValSamples": len(valIdx),
        "nTestSamples": len(testIdx),
        "splitStrategy": "temporal",
        "finalTrainLoss": float(history["trainLoss"][-1]) if history["trainLoss"] else None,
        "finalValLoss": float(history["valLoss"][-1]) if history["valLoss"] else None,
    }
    (outputDir / "trainReport.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return report


# ── CLI ─────────────────────────────────────────────────────

RUNNERS = {
    "endToEnd": runEndToEnd,
    "podRom": runPodRom,
    "pinn": runPinn,
    "gnn": runGnn,
}


def main() -> None:
    configPath = sys.argv[1] if len(sys.argv) > 1 else "training/config/trainEndToEnd.template.json"
    config = loadConfig(configPath)
    mode = config.get("mode", "endToEnd")
    runner = RUNNERS.get(mode)
    if runner is None:
        raise ValueError(f"unknown training mode: {mode}  (available: {list(RUNNERS.keys())})")
    report = runner(config)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
