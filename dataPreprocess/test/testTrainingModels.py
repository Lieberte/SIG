import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from training.endToEnd.model import bcToSurfaceModel
from training.gnn.model import gatBlock, gcnBlock, meshGnnEncoder, spatialGnnModel, temporalMeshGnn
from training.pinn.model import pinnModel
from training.pinn.physicsLoss import continuityLoss, fieldGradients
from training.podRom.model import latentDynamicsModel, podRomSurfaceModel
from training.shared.graphBuilder import buildEdgeAttr
from training.shared.normalizer import standardScaler
from training.shared.sequenceDataset import timeSeriesDataset
from training.shared.trainer import romTrainer


def testBcToSurfaceForward():
    model = bcToSurfaceModel(
        nInputFeatures=3, nOutputFeatures=10, hiddenSize=32, nLayers=1,
    )
    x = torch.randn(4, 8, 3)
    out = model(x)
    assert out.shape == (4, 10)


def testBcToSurfaceRollout():
    model = bcToSurfaceModel(
        nInputFeatures=2, nOutputFeatures=5, hiddenSize=16, nLayers=1,
    )
    initSeq = torch.randn(1, 4, 2)
    bcFuture = torch.randn(3, 2)
    preds = model.rollout(initSeq, bcFuture)
    assert preds.shape == (3, 5)


def testLatentDynamicsForward():
    model = latentDynamicsModel(
        nLatent=5, nBcFeatures=2, hiddenSize=32, nLayers=1,
    )
    x = torch.randn(4, 8, 7)
    out = model(x)
    assert out.shape == (4, 5)


def testLatentDynamicsRollout():
    model = latentDynamicsModel(
        nLatent=4, nBcFeatures=2, hiddenSize=16, nLayers=1,
    )
    initSeq = torch.randn(1, 5, 6)
    bcFuture = torch.randn(3, 2)
    preds = model.rollout(initSeq, bcFuture)
    assert preds.shape == (3, 4)


def testPodRomSurfaceModel():
    model = podRomSurfaceModel(
        nLatent=4, nBcFeatures=2, nSurfaceCells=20, hiddenSize=16, nLayers=1,
    )
    x = torch.randn(2, 6, 6)
    latentPred, surfacePred = model(x)
    assert latentPred.shape == (2, 4)
    assert surfacePred.shape == (2, 20)


def testPodRomSurfaceRollout():
    model = podRomSurfaceModel(
        nLatent=4, nBcFeatures=2, nSurfaceCells=20, hiddenSize=16, nLayers=1,
    )
    initSeq = torch.randn(1, 5, 6)
    bcFuture = torch.randn(3, 2)
    latentPreds, surfacePreds = model.rollout(initSeq, bcFuture)
    assert latentPreds.shape == (3, 4)
    assert surfacePreds.shape == (3, 20)


def testTimeSeriesDataset():
    inputSeq = np.random.randn(20, 3).astype(np.float32)
    targetSeq = np.random.randn(20, 5).astype(np.float32)
    ds = timeSeriesDataset(inputSeq, targetSeq, seqLen=5)
    assert len(ds) == 15
    x, y = ds[0]
    assert x.shape == (5, 3)
    assert y.shape == (5,)


def testTimeSeriesDatasetEmpty():
    inputSeq = np.random.randn(3, 2).astype(np.float32)
    targetSeq = np.random.randn(3, 4).astype(np.float32)
    ds = timeSeriesDataset(inputSeq, targetSeq, seqLen=10)
    assert len(ds) == 0


def testStandardScaler():
    np.random.seed(42)
    data = np.random.randn(100, 4)
    scaler = standardScaler()
    scaled = scaler.fitTransform(data)
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-10)
    recovered = scaler.inverseTransform(scaled)
    np.testing.assert_allclose(recovered, data, atol=1e-10)


def testStandardScalerSaveLoad():
    data = np.random.randn(50, 3)
    scaler = standardScaler().fit(data)
    params = scaler.saveParams()
    scaler2 = standardScaler().loadParams(params)
    np.testing.assert_array_equal(scaler.mean, scaler2.mean)
    np.testing.assert_array_equal(scaler.std, scaler2.std)


def testTrainerSmoke():
    model = bcToSurfaceModel(
        nInputFeatures=2, nOutputFeatures=3, hiddenSize=8, nLayers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lossFn = torch.nn.MSELoss()
    trainer = romTrainer(model, optimizer, lossFn)
    inputSeq = np.random.randn(30, 2).astype(np.float32)
    targetSeq = np.random.randn(30, 3).astype(np.float32)
    ds = timeSeriesDataset(inputSeq, targetSeq, seqLen=5)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    history = trainer.fit(loader, nEpochs=3, patience=0, verbose=False)
    assert len(history["trainLoss"]) == 3


def testPinnForward():
    model = pinnModel(nSpatialDim=2, nBcParams=1, nOutputFields=4, hiddenSize=16, nBlocks=2)
    coords = torch.randn(10, 2, requires_grad=True)
    t = torch.randn(10, 1, requires_grad=True)
    bc = torch.randn(10, 1)
    out = model(coords, t, bc)
    assert out.shape == (10, 4)


def testPinnFieldGradients():
    coords = torch.randn(5, 2, requires_grad=True)
    t = torch.randn(5, 1, requires_grad=True)
    u = (coords[:, 0:1] ** 2 + t)
    fields = {"u": u}
    grads = fieldGradients(fields, coords, t)
    assert "du_dx" in grads
    assert "du_dy" in grads
    assert "du_dt" in grads
    assert grads["du_dt"].shape == (5, 1)


def testPinnPredictFields():
    model = pinnModel(nSpatialDim=2, nBcParams=0, nOutputFields=3, hiddenSize=8, nBlocks=1)
    model.fieldNames = ["u", "v", "p"]
    coords = torch.randn(5, 2, requires_grad=True)
    t = torch.randn(5, 1, requires_grad=True)
    fields = model.predictFields(coords, t)
    assert set(fields.keys()) == {"u", "v", "p"}
    assert fields["u"].shape == (5, 1)


# ── PyG GNN tests ────────────────────────────────────────────

def testGcnBlockForward():
    block = gcnBlock(8, 8)
    x = torch.randn(6, 8)
    edgeIndex = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    out = block(x, edgeIndex)
    assert out.shape == (6, 8)


def testGatBlockForward():
    block = gatBlock(8, 8, nHeads=2)
    x = torch.randn(6, 8)
    edgeIndex = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    out = block(x, edgeIndex)
    assert out.shape == (6, 8)


def testGatBlockWithEdgeAttr():
    block = gatBlock(8, 8, nHeads=2, edgeDim=3)
    x = torch.randn(6, 8)
    edgeIndex = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edgeAttr = torch.randn(4, 3)
    out = block(x, edgeIndex, edgeAttr)
    assert out.shape == (6, 8)


def testMeshGnnEncoderGat():
    enc = meshGnnEncoder(nNodeFeatures=3, hiddenSize=8, nLayers=2, convType="gat", nHeads=2)
    x = torch.randn(10, 3)
    edgeIndex = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    out = enc(x, edgeIndex)
    assert out.shape == (10, 8)


def testMeshGnnEncoderGcn():
    enc = meshGnnEncoder(nNodeFeatures=3, hiddenSize=8, nLayers=2, convType="gcn")
    x = torch.randn(10, 3)
    edgeIndex = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    out = enc(x, edgeIndex)
    assert out.shape == (10, 8)


def testSpatialGnnForward():
    model = spatialGnnModel(nNodeFeatures=3, nOutputFeatures=2, hiddenSize=8, nLayers=2, convType="gat", nHeads=2)
    x = torch.randn(10, 3)
    edgeIndex = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    out = model(x, edgeIndex)
    assert out.shape == (10, 2)


def testSpatialGnnWithEdgeAttr():
    model = spatialGnnModel(nNodeFeatures=3, nOutputFeatures=2, hiddenSize=8, nLayers=2, convType="gat", nHeads=2, edgeDim=3)
    x = torch.randn(10, 3)
    edgeIndex = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    edgeAttr = torch.randn(5, 3)
    out = model(x, edgeIndex, edgeAttr)
    assert out.shape == (10, 2)


def testTemporalMeshGnnForward():
    model = temporalMeshGnn(
        nNodeFeatures=3, nOutputFeatures=3, hiddenSize=8,
        nConvLayers=2, convType="gat", nHeads=2, nGruLayers=1,
    )
    nodeFeatureSeq = torch.randn(2, 4, 6, 3)
    edgeIndex = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    out = model(nodeFeatureSeq, edgeIndex)
    assert out.shape == (2, 6, 3)


def testTemporalMeshGnnWithEdgeAttr():
    model = temporalMeshGnn(
        nNodeFeatures=3, nOutputFeatures=3, hiddenSize=8,
        nConvLayers=2, convType="gat", nHeads=2, nGruLayers=1, edgeDim=3,
    )
    nodeFeatureSeq = torch.randn(2, 4, 6, 3)
    edgeIndex = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    edgeAttr = torch.randn(6, 3)
    out = model(nodeFeatureSeq, edgeIndex, edgeAttr)
    assert out.shape == (2, 6, 3)


def testTemporalMeshGnnRollout():
    model = temporalMeshGnn(
        nNodeFeatures=3, nOutputFeatures=3, hiddenSize=8,
        nConvLayers=2, convType="gat", nHeads=2, nGruLayers=1,
    )
    initSeq = torch.randn(1, 4, 6, 3)
    edgeIndex = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    preds = model.rollout(initSeq, edgeIndex, nSteps=3)
    assert preds.shape == (3, 6, 3)


def testTemporalGcnForward():
    model = temporalMeshGnn(
        nNodeFeatures=3, nOutputFeatures=3, hiddenSize=8,
        nConvLayers=2, convType="gcn", nGruLayers=1,
    )
    nodeFeatureSeq = torch.randn(2, 4, 6, 3)
    edgeIndex = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    out = model(nodeFeatureSeq, edgeIndex)
    assert out.shape == (2, 6, 3)


def testBuildEdgeAttr():
    edgeType = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    oneHot = buildEdgeAttr(edgeType, nTypes=3)
    assert oneHot.shape == (5, 3)
    np.testing.assert_array_equal(oneHot[0], [1, 0, 0])
    np.testing.assert_array_equal(oneHot[1], [0, 1, 0])
    np.testing.assert_array_equal(oneHot[2], [0, 0, 1])


if __name__ == "__main__":
    testBcToSurfaceForward()
    testBcToSurfaceRollout()
    testLatentDynamicsForward()
    testLatentDynamicsRollout()
    testPodRomSurfaceModel()
    testPodRomSurfaceRollout()
    testTimeSeriesDataset()
    testTimeSeriesDatasetEmpty()
    testStandardScaler()
    testStandardScalerSaveLoad()
    testTrainerSmoke()
    testPinnForward()
    testPinnFieldGradients()
    testPinnPredictFields()
    testGcnBlockForward()
    testGatBlockForward()
    testGatBlockWithEdgeAttr()
    testMeshGnnEncoderGat()
    testMeshGnnEncoderGcn()
    testSpatialGnnForward()
    testSpatialGnnWithEdgeAttr()
    testTemporalMeshGnnForward()
    testTemporalMeshGnnWithEdgeAttr()
    testTemporalMeshGnnRollout()
    testTemporalGcnForward()
    testBuildEdgeAttr()
    print("all training model tests passed")
