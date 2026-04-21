#!/usr/bin/env python3
"""
Test script for shared infrastructure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "dataPreprocess"))

import numpy as np
import json
import tempfile
import shutil

from training.shared.dataSplit import (
    temporal_split,
    case_split,
    random_split,
    split_dataset,
    DataSplitter,
)
from training.shared.evaluationMetrics import (
    compute_metrics,
    compute_rollout_error,
    evaluate_predictions,
    PhysicsConsistency,
)
from training.shared.trainingLogger import ExperimentLogger, RunConfig, EpochLog, MetricsTracker


def test_data_split():
    print("=== Testing data split utilities ===")

    n_samples = 100
    train_idx, val_idx, test_idx = temporal_split(n_samples, val_ratio=0.2, test_ratio=0.1)
    assert len(train_idx) == 70
    assert len(val_idx) == 20
    assert len(test_idx) == 10
    assert np.all(np.diff(train_idx) >= 0)
    assert np.all(np.diff(val_idx) >= 0)
    assert np.all(np.diff(test_idx) >= 0)
    assert int(np.min(val_idx)) == 70
    assert int(np.min(test_idx)) == 90
    print("  temporal_split: PASS")

    case_ids = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    train_idx, val_idx, test_idx = case_split(
        case_ids,
        test_cases=[1],
        val_cases=[2],
    )
    assert set(case_ids[train_idx]) == {3}
    assert set(case_ids[val_idx]) == {2}
    assert set(case_ids[test_idx]) == {1}
    print("  case_split: PASS")

    train_idx, val_idx, test_idx = random_split(n_samples, val_ratio=0.2, test_ratio=0.1, seed=999)
    assert len(train_idx) + len(val_idx) + len(test_idx) == n_samples
    assert len(set(train_idx) & set(val_idx)) == 0
    print("  random_split: PASS")

    train_idx, val_idx, test_idx = split_dataset(
        n_samples=100, strategy="temporal", val_ratio=0.1, test_ratio=0.1
    )
    assert len(train_idx) == 80
    assert len(val_idx) == 10
    assert len(test_idx) == 10
    print("  split_dataset temporal: PASS")

    case_ids = np.repeat(np.arange(10), 10)
    train_idx, val_idx, test_idx = split_dataset(
        n_samples=100,
        strategy="case",
        case_ids=case_ids,
        val_ratio=0.2,
        test_ratio=0.1,
    )
    assert len(train_idx) + len(val_idx) + len(test_idx) == n_samples
    print("  split_dataset case: PASS")

    ds = DataSplitter(strategy="temporal", val_ratio=0.1, test_ratio=0.1)
    ds.fit(n_samples=200)
    assert len(ds.train_idx) == 160  # 200 - 20(val) - 20(test) = 160
    print("  DataSplitter: PASS")


def test_evaluation_metrics():
    print("=== Testing evaluation metrics ===")
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([1.1, 2.0, 3.0, 4.1])

    metrics = compute_metrics(pred, target)
    # errors: [-0.1, 0, 0, -0.1] -> mse = 0.02/4 = 0.005 -> rmse = sqrt(0.005)
    expected_rmse = np.sqrt(0.005)
    assert abs(metrics.rmse - expected_rmse) < 1e-12
    print("  compute_metrics: PASS")

    pred_2d = np.array([[1, 2], [3, 4]])
    target_2d = np.array([[1.1, 1.9], [3.0, 4.1]])
    metrics = compute_metrics(pred_2d, target_2d, field_names=["T", "U"])
    assert "T" in metrics.per_field
    assert "U" in metrics.per_field
    print("  multi-field metrics: PASS")

    predictions = np.random.randn(5, 3, 4)
    targets = np.random.randn(5, 3, 4)
    r = compute_rollout_error(predictions, targets, seq_len=2, n_steps=3)
    assert "1_step_rmse" in r and "5_step_rmse" in r
    print("  compute_rollout_error: PASS")

    pc = PhysicsConsistency()
    rho = np.ones(10)
    div = np.ones(10) * 0.01
    mc_res = pc.mass_conservation_residual(rho, div)
    assert mc_res >= 0
    print("  PhysicsConsistency: PASS")

    full_report = evaluate_predictions(pred, target, field_names=["x"])
    assert "metrics" in full_report
    print("  evaluate_predictions: PASS")


def test_logger():
    print("=== Testing training logger ===")
    tmpdir = tempfile.mkdtemp()
    try:
        config = RunConfig(
            mode="test",
            experiment_name="exp_001",
            seed=123,
            device="cpu",
            hyperparameters={"lr": 1e-3},
            data_split={"val_ratio": 0.2},
        )

        logger = ExperimentLogger(
            output_dir=tmpdir,
            mode="test",
            experiment_name="exp_001",
            seed=123,
        )
        logger.start_run(config)

        for e in range(3):
            log = EpochLog(
                epoch=e,
                train_loss=1.0 / (e + 1),
                val_loss=0.5 / (e + 1),
                epoch_time=0.1,
                lr=1e-3,
                extra={"grad_norm": 0.5},
            )
            logger.log_epoch(log)

        metrics_tracker = MetricsTracker()
        for e in range(5):
            metrics_tracker.update(train_loss=1.0 / (e + 1), val_loss=0.5 / (e + 1), epoch=e)
        metrics_tracker.save(Path(tmpdir) / "metrics")

        logger.finish_run(final_metrics={"custom": 42})

        summary_path = Path(tmpdir) / "run_summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["experiment_name"] == "exp_001"
        print("  ExperimentLogger: PASS")

        assert (Path(tmpdir) / "metrics" / "train_losses.npy").exists()
        assert (Path(tmpdir) / "metrics" / "metrics_summary.json").exists()
        print("  MetricsTracker: PASS")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_data_split()
    print()
    test_evaluation_metrics()
    print()
    test_logger()
    print()
    print("All shared infrastructure tests passed.")
