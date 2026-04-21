"""
Unified evaluation metrics for ROM training.

Provides consistent evaluation across different training modes:
- POD-ROM, GNN, PINN, end-to-end

Metrics:
- RMSE, MAE, R², relative L2, max error
- Per-field and per-region statistics
- Rollout error for autoregressive models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    rmse: float
    mae: float
    r2: float
    rel_l2: float
    max_error: float
    per_field: dict[str, dict[str, float]] = field(default_factory=dict)
    per_region: dict[str, dict[str, float]] = field(default_factory=dict)
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "rel_l2": self.rel_l2,
            "max_error": self.max_error,
            "breakdown": self.breakdown,
        }
        if self.per_field:
            result["per_field"] = self.per_field
        if self.per_region:
            result["per_region"] = self.per_region
        return result

    def __str__(self) -> str:
        lines = [
            f"RMSE: {self.rmse:.6e}",
            f"MAE:  {self.mae:.6e}",
            f"R²:   {self.r2:.6f}",
            f"rel L2: {self.rel_l2:.6f}",
            f"max error: {self.max_error:.6e}",
        ]
        if self.breakdown:
            lines.append("\nBreakdown:")
            for k, v in self.breakdown.items():
                lines.append(f"  {k}: {v:.6e}")
        if self.per_field:
            lines.append("\nPer-field:")
            for field_name, field_metrics in self.per_field.items():
                lines.append(f"  {field_name}: RMSE={field_metrics.get('rmse', np.nan):.6e}, "
                           f"MAE={field_metrics.get('mae', np.nan):.6e}, "
                           f"R²={field_metrics.get('r2', np.nan):.6f}")
        if self.per_region:
            lines.append("\nPer-region:")
            for region_name, region_metrics in self.per_region.items():
                lines.append(f"  {region_name}: RMSE={region_metrics.get('rmse', np.nan):.6e}, "
                           f"MAE={region_metrics.get('mae', np.nan):.6e}")
        return "\n".join(lines)


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((pred - target) ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(pred - target))


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """R² score."""
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
    """Relative L2 error."""
    norm_pred = np.linalg.norm(pred)
    norm_target = np.linalg.norm(target)
    if norm_target < 1e-12:
        return 0.0
    return np.linalg.norm(pred - target) / norm_target


def compute_max_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Maximum absolute error."""
    return np.max(np.abs(pred - target))


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    field_names: list[str] | None = None,
    region_masks: dict[str, np.ndarray] | None = None,
) -> EvaluationMetrics:
    """
    Compute comprehensive evaluation metrics.

    Args:
        pred: Predicted values, shape (n_samples, n_features) or (n_samples,)
        target: Ground truth values, same shape as pred
        field_names: Names for each feature (for per-field breakdown)
        region_masks: Dict of region name -> boolean mask for per-region breakdown

    Returns:
        EvaluationMetrics object
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
        target = target.reshape(-1, 1)

    n_samples, n_features = pred.shape

    # Global metrics
    rmse = compute_rmse(pred, target)
    mae = compute_mae(pred, target)
    r2 = compute_r2(pred, target)
    rel_l2 = compute_rel_l2(pred, target)
    max_error = compute_max_error(pred, target)

    breakdown = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rel_l2": rel_l2,
        "max_error": max_error,
    }

    per_field: dict[str, dict[str, float]] = {}
    if field_names is not None:
        for i, name in enumerate(field_names):
            if i >= n_features:
                break
            p = pred[:, i]
            t = target[:, i]
            per_field[name] = {
                "rmse": compute_rmse(p, t),
                "mae": compute_mae(p, t),
                "r2": compute_r2(p, t),
                "rel_l2": compute_rel_l2(p, t),
                "max_error": compute_max_error(p, t),
            }
            breakdown[f"rmse_{name}"] = per_field[name]["rmse"]
            breakdown[f"mae_{name}"] = per_field[name]["mae"]
            breakdown[f"r2_{name}"] = per_field[name]["r2"]

    per_region: dict[str, dict[str, float]] = {}
    if region_masks is not None:
        for region_name, mask in region_masks.items():
            p = pred[mask]
            t = target[mask]
            if p.size > 0 and t.size > 0:
                per_region[region_name] = {
                    "rmse": compute_rmse(p, t),
                    "mae": compute_mae(p, t),
                    "r2": compute_r2(p, t),
                    "rel_l2": compute_rel_l2(p, t),
                    "max_error": compute_max_error(p, t),
                }
                breakdown[f"rmse_{region_name}"] = per_region[region_name]["rmse"]
                breakdown[f"mae_{region_name}"] = per_region[region_name]["mae"]

    return EvaluationMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        rel_l2=rel_l2,
        max_error=max_error,
        per_field=per_field,
        per_region=per_region,
        breakdown=breakdown,
    )


def compute_rollout_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
    n_steps: int,
) -> dict[str, float]:
    """
    Compute error for autoregressive rollout.

    Args:
        predictions: Array of shape (n_samples, n_steps, n_features) or (n_samples, n_features)
        targets: Ground truth, same shape as predictions
        seq_len: Length of initial conditioning window
        n_steps: Number of rollout steps

    Returns:
        Dict with 1-step, 5-step, 10-step errors
    """
    if predictions.ndim == 2:
        # Single step predictions
        if seq_len == 1 and n_steps == 1:
            metrics = compute_metrics(predictions, targets)
            return {
                "1_step_rmse": metrics.rmse,
                "5_step_rmse": np.nan,
                "10_step_rmse": np.nan,
            }
        else:
            raise ValueError("For single-step predictions, seq_len and n_steps must both be 1")

    # Multi-step predictions: (n_samples, n_steps, n_features)
    results = {}
    for k in [1, 5, 10]:
        if k > n_steps:
            results[f"{k}_step_rmse"] = np.nan
            continue

        step_errors = []
        for step in range(k):
            pred_step = predictions[:, step, :]
            target_step = targets[:, step, :]
            metrics = compute_metrics(pred_step, target_step)
            step_errors.append(metrics.rmse)

        results[f"{k}_step_rmse"] = float(np.mean(step_errors))

    return results


class PhysicsConsistency:
    """Physics-based consistency checks for CFD surrogates."""

    @staticmethod
    def mass_conservation_residual(
        rho: np.ndarray,
        divergence: np.ndarray,
    ) -> float:
        """
        Check mass conservation residual.

        Args:
            rho: Density field
            divergence: Divergence of velocity field

        Returns:
            Mean squared residual
        """
        return np.mean((rho * divergence) ** 2)

    @staticmethod
    def energy_conservation_residual(
        energy_in: np.ndarray,
        energy_out: np.ndarray,
    ) -> float:
        """
        Check energy conservation residual.

        Args:
            energy_in: Incoming energy flux
            energy_out: Outgoing energy flux

        Returns:
            Mean squared residual
        """
        return np.mean((energy_in - energy_out) ** 2)

    @staticmethod
    def boundary_compliance(
        predicted: np.ndarray,
        boundary_values: np.ndarray,
        tolerance: float = 1e-3,
    ) -> dict[str, float]:
        """
        Check boundary condition compliance.

        Args:
            predicted: Predicted boundary values
            boundary_values: True boundary values
            tolerance: Relative tolerance

        Returns:
            Dict with compliance metrics
        """
        rel_error = np.abs(predicted - boundary_values) / (np.abs(boundary_values) + 1e-12)
        max_error = np.max(rel_error)
        mean_error = np.mean(rel_error)
        compliant = np.sum(rel_error < tolerance) / len(rel_error)

        return {
            "max_relative_error": float(max_error),
            "mean_relative_error": float(mean_error),
            "compliance_rate": float(compliant),
            "tolerance": tolerance,
        }

    @staticmethod
    def field_bounds_compliance(
        predicted: np.ndarray,
        bounds: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        """
        Check if predicted fields stay within physical bounds.

        Args:
            predicted: Predicted field values
            bounds: Dict of field name -> (min, max) bounds

        Returns:
            Dict with violation statistics
        """
        violations = {}
        for field_name, (min_val, max_val) in bounds.items():
            if field_name not in predicted:
                continue
            field_data = predicted[field_name]
            out_of_bounds = (field_data < min_val) | (field_data > max_val)
            n_violations = np.sum(out_of_bounds)
            violation_rate = n_violations / len(field_data) if len(field_data) > 0 else 0.0
            violations[field_name] = {
                "n_violations": int(n_violations),
                "violation_rate": float(violation_rate),
                "min_value": float(np.min(field_data)),
                "max_value": float(np.max(field_data)),
                "bounds": (float(min_val), float(max_val)),
            }
        return violations


def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    field_names: list[str] | None = None,
    region_masks: dict[str, np.ndarray] | None = None,
    rollout_info: dict | None = None,
    physics_checks: dict | None = None,
) -> dict[str, Any]:
    """
    Comprehensive prediction evaluation.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        field_names: Names for each feature
        region_masks: Dict of region masks
        rollout_info: Info for rollout evaluation
        physics_checks: Physics consistency checks

    Returns:
        Complete evaluation report
    """
    metrics = compute_metrics(predictions, targets, field_names, region_masks)
    report = {
        "metrics": metrics.to_dict(),
        "predictions_shape": predictions.shape,
        "targets_shape": targets.shape,
    }

    if rollout_info is not None:
        rollout_errors = compute_rollout_error(
            predictions,
            targets,
            rollout_info.get("seq_len", 1),
            rollout_info.get("n_steps", 1),
        )
        report["rollout_errors"] = rollout_errors

    if physics_checks is not None:
        physics_report = {}
        for check_name, check_fn in physics_checks.items():
            result = check_fn(predictions, targets)
            physics_report[check_name] = result
        report["physics_checks"] = physics_report

    return report
