#!/usr/bin/env python3
"""
Integration test for all training pipelines with synthetic data.
Tests end-to-end (TCN/MLP), GNN (GRU/TCN), POD-ROM, and PINN.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "dataPreprocess"))

import numpy as np
import torch
import tempfile
import shutil


def test_rnn_model():
    """Test hand-written RNN with explicit state equations."""
    print("=== Testing hand-written RNN model ===")
    from training.endToEnd.model import rnnStateUpdater, rnnModel

    batch, seqLen, nInput, nOutput, hidden = 2, 5, 4, 10, 32

    # Test core state updater
    core = rnnStateUpdater(nInput, hidden, nOutput, activation="tanh", dropout=0.1)
    x = torch.randn(batch, seqLen, nInput)
    outputs, h_final = core(x)
    assert outputs.shape == (batch, seqLen, nOutput)
    assert h_final.shape == (batch, hidden)
    print("  rnnStateUpdater forward: PASS")

    # Verify explicit state equation (use no dropout for verification)
    core_no_dropout = rnnStateUpdater(nInput, hidden, nOutput, activation="tanh", dropout=0.0)
    core_no_dropout.W_hh.data = core.W_hh.data.clone()
    core_no_dropout.W_xh.data = core.W_xh.data.clone()
    core_no_dropout.b_h.data = core.b_h.data.clone()
    core_no_dropout.W_hy.data = core.W_hy.data.clone()
    core_no_dropout.b_y.data = core.b_y.data.clone()

    h0 = core_no_dropout.reset_state(batch, x.device)
    y1, h1 = core_no_dropout.step(x[:, 0], h0)
    expected_h1 = torch.tanh(h0 @ core_no_dropout.W_hh.T + x[:, 0] @ core_no_dropout.W_xh.T + core_no_dropout.b_h)
    assert torch.allclose(h1, expected_h1, atol=1e-6), "State equation mismatch"
    print("  State equation (h_t = act(W_hh·h + W_xh·x + b)): PASS")

    # Test wrapper model
    model = rnnModel(nInput, nOutput, hiddenSize=hidden)
    y = model(x)
    assert y.shape == (batch, nOutput)
    print("  rnnModel forward: PASS")

    # Test rollout
    bcHistory = torch.randn(1, seqLen, nInput)
    bcFuture = torch.randn(3, nInput)
    rollout = model.rollout(bcHistory, bcFuture)
    assert rollout.shape[0] == 3
    print("  rnnModel rollout: PASS")
    print()


def test_end_to_end_tcn():
    """Test TCN-based end-to-end model."""
    print("=== Testing end-to-end TCN model ===")
    from training.endToEnd.model import bcToSurfaceModel

    batch, seqLen, nBc, nOutput = 2, 5, 4, 10
    model = bcToSurfaceModel(
        nInputFeatures=nBc,
        nOutputFeatures=nOutput,
        hiddenSize=32,
        nLayers=2,
        dropout=0.1,
        encoderType="tcn",
        kernelSize=3,
    )

    x = torch.randn(batch, seqLen, nBc)
    y = model(x)
    assert y.shape == (batch, nOutput), f"Expected {(batch, nOutput)}, got {y.shape}"
    print("  TCN forward: PASS")

    # Test rollout
    bcHistory = torch.randn(1, seqLen, nBc)
    bcFuture = torch.randn(3, nBc)
    rollout = model.rollout(bcHistory, bcFuture)
    assert rollout.shape[0] == 3
    print("  TCN rollout: PASS")
    print()


def test_end_to_end_mlp():
    """Test MLP-based end-to-end model (no time window)."""
    print("=== Testing end-to-end MLP model ===")
    from training.endToEnd.model import mlpModel

    batch, nBc, nOutput = 8, 4, 10
    model = mlpModel(
        nInputFeatures=nBc,
        nOutputFeatures=nOutput,
        hiddenSize=64,
        nHiddenLayers=3,
        dropout=0.1,
        activation="relu",
    )

    x = torch.randn(batch, nBc)
    y = model(x)
    assert y.shape == (batch, nOutput), f"Expected {(batch, nOutput)}, got {y.shape}"
    print("  MLP forward (1D): PASS")

    # Test with 3D input (backward compatible)
    x3d = torch.randn(batch, 1, nBc)
    y3d = model(x3d)
    assert y3d.shape == (batch, nOutput)
    print("  MLP forward (3D compat): PASS")
    print()


def test_gnn_tcn():
    """Test GNN with TCN temporal encoding."""
    print("=== Testing GNN TCN model ===")
    from training.gnn.model import temporalMeshGnn

    batch, seqLen, nNodes, nFeat, nOut = 2, 5, 10, 4, 4
    model = temporalMeshGnn(
        nNodeFeatures=nFeat,
        nOutputFeatures=nOut,
        hiddenSize=32,
        nConvLayers=2,
        convType="gcn",
        nGruLayers=2,
        dropout=0.1,
        temporalType="tcn",
        kernelSize=3,
    )

    nodeFeatures = torch.randn(batch, seqLen, nNodes, nFeat)
    edgeIndex = torch.randint(0, nNodes, (2, 20))

    output = model(nodeFeatures, edgeIndex)
    assert output.shape == (batch, nNodes, nOut), f"Expected {(batch, nNodes, nOut)}, got {output.shape}"
    print("  GNN TCN forward: PASS")
    print()


def test_gnn_gru():
    """Test GNN with GRU temporal encoding."""
    print("=== Testing GNN GRU model ===")
    from training.gnn.model import temporalMeshGnn

    batch, seqLen, nNodes, nFeat, nOut = 2, 5, 10, 4, 4
    model = temporalMeshGnn(
        nNodeFeatures=nFeat,
        nOutputFeatures=nOut,
        hiddenSize=32,
        nConvLayers=2,
        convType="gcn",
        nGruLayers=1,
        dropout=0.1,
        temporalType="gru",
    )

    nodeFeatures = torch.randn(batch, seqLen, nNodes, nFeat)
    edgeIndex = torch.randint(0, nNodes, (2, 20))

    output = model(nodeFeatures, edgeIndex)
    assert output.shape == (batch, nNodes, nOut)
    print("  GNN GRU forward: PASS")
    print()


def test_pod_rom():
    """Test POD-ROM pipeline."""
    print("=== Testing POD-ROM pipeline ===")
    from training.shared.podBasis import podBasis
    from training.shared.dataSplit import temporal_split

    nCells = 100
    nFields = 4
    nSnapshots = 30
    matrix = np.random.randn(nCells * nFields, nSnapshots)

    # Temporal split
    trainIdx, valIdx, testIdx = temporal_split(nSnapshots, val_ratio=0.2)
    assert len(trainIdx) + len(valIdx) == nSnapshots

    # POD fit on train only
    trainMatrix = matrix[:, trainIdx]
    pod = podBasis(nModes=10)
    pod.fit(trainMatrix)

    # Energy report
    report = pod.energyReport()
    assert "nModesUsed" in report
    assert report["truncatedEnergy"] <= 1.0

    # Encode all snapshots
    latent = pod.encode(matrix)
    assert latent.shape == (10, nSnapshots)

    # Reconstruction
    reconstructed = pod.decode(latent)
    assert reconstructed.shape == matrix.shape

    print("  POD-ROM temporal split + fit: PASS")
    print("  POD-ROM energy report: PASS")
    print("  POD-ROM encode/decode: PASS")
    print()


def test_pinn():
    """Test PINN model and loss."""
    print("=== Testing PINN pipeline ===")
    from training.pinn.model import pinnModel, pinnLossComposer

    device = "cpu"
    model = pinnModel(
        nSpatialDim=3,
        nBcParams=0,
        nOutputFields=6,
        hiddenSize=32,
        nBlocks=2,
    ).to(device)

    # Forward pass
    coords = torch.randn(100, 3, device=device)
    t = torch.randn(100, 1, device=device)
    output = model(coords, t)
    assert output.shape == (100, 6)
    print("  PINN forward: PASS")

    # Physics loss
    physicsParams = {"mu": 1e-3, "cp": 1000.0, "k": 0.6}
    composer = pinnLossComposer(model, physicsParams)

    lossDict = composer.physicsLoss(coords, t)
    assert isinstance(lossDict, dict)
    print("  PINN physics loss: PASS")
    print()


def test_integration():
    """Full integration test with synthetic pipeline data."""
    print("=== Integration test: full pipeline flow ===")
    from training.shared.dataSplit import temporal_split
    from training.shared.evaluationMetrics import compute_metrics
    from training.shared.normalizer import standardScaler

    # Simulate end-to-end training data
    nSnapshots = 20
    nBc = 4
    nOutput = 10
    seqLen = 3

    bcInput = np.random.randn(nSnapshots, nBc)
    target = np.random.randn(nSnapshots, nOutput)

    # Normalize
    bcScaler = standardScaler()
    targetScaler = standardScaler()
    bcNorm = bcScaler.fitTransform(bcInput)
    targetNorm = targetScaler.fitTransform(target)

    # Temporal split
    trainIdx, valIdx, testIdx = temporal_split(nSnapshots, 0.2)
    nTrain = len(trainIdx)
    assert nTrain == nSnapshots - int(nSnapshots * 0.2)

    # MLP training simulation
    from training.endToEnd.model import mlpModel
    model = mlpModel(nBc, nOutput, hiddenSize=32, nHiddenLayers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainLosses = []
    for epoch in range(10):
        model.train()
        x = torch.from_numpy(bcNorm[trainIdx].astype(np.float32))
        y = torch.from_numpy(targetNorm[trainIdx].astype(np.float32))
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainLosses.append(loss.item())

    assert trainLosses[-1] < trainLosses[0] or trainLosses[-1] == trainLosses[0]
    print(f"  Training loss: {trainLosses[0]:.6f} → {trainLosses[-1]:.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        valPred = model(torch.from_numpy(bcNorm[valIdx].astype(np.float32)))
        valTarget = torch.from_numpy(targetNorm[valIdx].astype(np.float32))
        valLoss = torch.nn.functional.mse_loss(valPred, valTarget).item()
    print(f"  Val loss: {valLoss:.6f}")

    metrics = compute_metrics(valPred.numpy(), valTarget.numpy())
    print(f"  Val metrics: RMSE={metrics.rmse:.6f}, R2={metrics.r2:.6f}")

    print("  Integration test: PASS")
    print()


if __name__ == "__main__":
    test_rnn_model()
    test_end_to_end_tcn()
    test_end_to_end_mlp()
    test_gnn_tcn()
    test_gnn_gru()
    test_pod_rom()
    test_pinn()
    test_integration()
    print("All integration tests passed!")
