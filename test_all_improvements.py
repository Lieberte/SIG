#!/usr/bin/env python3
"""
Comprehensive test for all training improvements.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "dataPreprocess"))

import numpy as np
import json
import tempfile
import shutil


def test_pod_rom_preprocess():
    print("=== Testing POD-ROM preprocess ===")
    from training.podRom.preprocess import PodRomPreprocessor, PodRomDataset, computeBcStatistics
    from training.shared.podBasis import podBasis

    # Create synthetic data
    nCells = 100
    nFields = 4
    nSnapshots = 50
    matrix = np.random.randn(nCells * nFields, nSnapshots)

    schema = {
        "nCells": nCells,
        "fieldOrder": ["u", "v", "p", "T"],
    }
    topologyMeta = {"faceZones": []}

    # Test podBasis energy report
    pod = podBasis(nModes=10)
    pod.fit(matrix)
    report = pod.energyReport()
    assert "nModesUsed" in report
    assert "truncatedEnergy" in report
    assert report["truncatedEnergy"] <= 1.0
    print("  podBasis energyReport: PASS")

    # Test save/load
    tmpdir = tempfile.mkdtemp()
    try:
        pod.save(Path(tmpdir))
        loaded = podBasis.load(Path(tmpdir))
        assert loaded.nModes == pod.nModes
        print("  podBasis save/load: PASS")
    finally:
        shutil.rmtree(tmpdir)

    print()


def test_gnn_model():
    print("=== Testing GNN model ===")
    import torch
    from training.gnn.model import temporalMeshGnn, meshGnnEncoder

    # Test model creation
    model = temporalMeshGnn(
        nNodeFeatures=4,
        nOutputFeatures=4,
        hiddenSize=32,
        nConvLayers=2,
        convType="gat",
        nHeads=2,
    )

    # Test forward pass
    batch, seqLen, nNodes, nFeat = 2, 5, 10, 4
    nodeFeatures = torch.randn(batch, seqLen, nNodes, nFeat)
    edgeIndex = torch.randint(0, nNodes, (2, 20))

    output = model(nodeFeatures, edgeIndex)
    assert output.shape == (batch, nNodes, 4)
    print("  temporalMeshGnn forward: PASS")

    # Test rollout
    initSeq = torch.randn(1, seqLen, nNodes, nFeat)
    preds = model.rollout(initSeq, edgeIndex, nSteps=3)
    assert preds.shape[0] == 3
    print("  temporalMeshGnn rollout: PASS")

    print()


def test_pinn_preprocess():
    print("=== Testing PINN preprocess ===")
    from training.pinn.preprocess import PinnPreprocessor, PinnDataset, preparePinnData

    # Create synthetic data
    nCells = 100
    nFields = 4
    nSnapshots = 20
    matrix = np.random.randn(nCells * nFields, nSnapshots)

    schema = {
        "nCells": nCells,
        "fieldOrder": ["u", "v", "p", "T"],
    }
    topologyMeta = {"faceZones": []}

    # Test preprocessor
    tmpdir = tempfile.mkdtemp()
    try:
        dataset = preparePinnData(
            snapshotMatrix=matrix,
            schema=schema,
            topologyMeta=topologyMeta,
            casePath=Path(tmpdir),  # dummy path
            fieldNames=["u", "v", "p", "T"],
            nCollocationPoints=100,
            nBoundaryPoints=50,
            nInitialPoints=50,
            nDataPoints=100,
        )

        assert len(dataset.collocationCoords) == 100
        assert len(dataset.boundaryCoords) <= 50
        assert len(dataset.initialCoords) == 50
        assert len(dataset.dataCoords) == 100
        print("  PinnDataset creation: PASS")

        # Test save/load
        dataset.save(Path(tmpdir))
        loaded = PinnDataset.load(Path(tmpdir))
        assert len(loaded.collocationCoords) == len(dataset.collocationCoords)
        print("  PinnDataset save/load: PASS")
    finally:
        shutil.rmtree(tmpdir)

    print()


def test_data_split():
    print("=== Testing data split ===")
    from training.shared.dataSplit import temporal_split, case_split, random_split, DataSplitter

    # Temporal split
    train, val, test = temporal_split(100, 0.2, 0.1)
    assert len(train) == 70
    assert len(val) == 20
    assert len(test) == 10
    assert np.max(train) < np.min(val)
    assert np.max(val) < np.min(test)
    print("  temporal_split: PASS")

    # Case split
    caseIds = np.repeat(np.arange(5), 10)
    train, val, test = case_split(caseIds, test_cases=[0], val_cases=[1])
    assert set(caseIds[train]) == {2, 3, 4}
    print("  case_split: PASS")

    # DataSplitter
    splitter = DataSplitter(strategy="temporal", val_ratio=0.2)
    splitter.fit(100)
    info = splitter.get_split_info()
    assert info["strategy"] == "temporal"
    print("  DataSplitter: PASS")

    print()


def test_evaluation_metrics():
    print("=== Testing evaluation metrics ===")
    from training.shared.evaluationMetrics import compute_metrics, compute_rollout_error, PhysicsConsistency

    pred = np.random.randn(100, 4)
    target = np.random.randn(100, 4)

    metrics = compute_metrics(pred, target, field_names=["u", "v", "p", "T"])
    assert metrics.rmse >= 0
    assert metrics.mae >= 0
    assert metrics.r2 <= 1.0
    assert "u" in metrics.per_field
    print("  compute_metrics: PASS")

    # Rollout error
    preds = np.random.randn(10, 5, 4)
    targets = np.random.randn(10, 5, 4)
    rollout = compute_rollout_error(preds, targets, seq_len=2, n_steps=5)
    assert "1_step_rmse" in rollout
    print("  compute_rollout_error: PASS")

    # Physics consistency
    pc = PhysicsConsistency()
    rho = np.ones(10)
    div = np.ones(10) * 0.01
    res = pc.mass_conservation_residual(rho, div)
    assert res >= 0
    print("  PhysicsConsistency: PASS")

    print()


def test_training_logger():
    print("=== Testing training logger ===")
    from training.shared.trainingLogger import ExperimentLogger, RunConfig, EpochLog, MetricsTracker

    tmpdir = tempfile.mkdtemp()
    try:
        logger = ExperimentLogger(
            output_dir=tmpdir,
            mode="test",
            experiment_name="test_exp",
        )

        config = RunConfig(mode="test", seed=42)
        logger.start_run(config)

        for e in range(5):
            log = EpochLog(epoch=e, train_loss=1.0 / (e + 1), val_loss=0.5 / (e + 1))
            logger.log_epoch(log)

        logger.finish_run()

        assert (Path(tmpdir) / "run_summary.json").exists()
        assert (Path(tmpdir) / "epoch_logs.json").exists()
        print("  ExperimentLogger: PASS")

        # MetricsTracker
        tracker = MetricsTracker()
        for e in range(5):
            tracker.update(train_loss=1.0 / (e + 1), val_loss=0.5 / (e + 1), epoch=e)
        tracker.save(Path(tmpdir))
        assert (Path(tmpdir) / "train_losses.npy").exists()
        print("  MetricsTracker: PASS")
    finally:
        shutil.rmtree(tmpdir)

    print()


if __name__ == "__main__":
    test_data_split()
    test_evaluation_metrics()
    test_training_logger()
    test_pod_rom_preprocess()
    test_gnn_model()
    test_pinn_preprocess()
    print("All tests passed!")
