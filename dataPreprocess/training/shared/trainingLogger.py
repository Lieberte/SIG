"""
Training logger for ROM experiments.

Supports:
- Local file logging with JSON metrics
- Optional tensorboard / wandb integration
- Reproducibility management
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RunConfig:
    """Configuration for a training run."""

    mode: str
    experiment_name: str | None = None
    seed: int = 42
    device: str = "cpu"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    data_split: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpochLog:
    """Log entry for a single epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None = None
    test_loss: float | None = None
    epoch_time: float | None = None
    lr: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = asdict(self)
        return result


class ExperimentLogger:
    """
    Manages experiment logging and configuration persistence.
    """

    def __init__(
        self,
        output_dir: str | Path,
        mode: str,
        experiment_name: str | None = None,
        seed: int = 42,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.seed = seed
        self.experiment_name = experiment_name or self._default_name()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_config: RunConfig | None = None
        self.epoch_logs: list[EpochLog] = []
        self.start_time: float | None = None
        self._tensorboard_writer = None
        self._wandb_run = None

        if use_tensorboard:
            self._init_tensorboard()

        if use_wandb:
            self._init_wandb(wandb_project, wandb_entity)

    def _default_name(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.mode}_{ts}"

    def _init_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = self.output_dir / "tensorboard" / self.experiment_name
            self._tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            pass

    def _init_wandb(self, project: str | None, entity: str | None) -> None:
        try:
            import wandb

            wandb.init(
                project=project or "rom-training",
                entity=entity,
                name=self.experiment_name,
                dir=str(self.output_dir),
                config=self.run_config.to_dict() if self.run_config else {},
            )
            self._wandb_run = wandb
        except ImportError:
            pass

    def start_run(self, config: RunConfig) -> None:
        """Mark the start of a training run."""
        self.run_config = config
        self.start_time = time.time()

        config_path = self.output_dir / "run_config.json"
        config_path.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        if self._wandb_run:
            self._wandb_run.config.update(config.to_dict())

    def log_epoch(self, log: EpochLog) -> None:
        """Log metrics for an epoch."""
        self.epoch_logs.append(log)

        if self._tensorboard_writer is not None:
            writer = self._tensorboard_writer
            writer.add_scalar("loss/train", log.train_loss, log.epoch)
            if log.val_loss is not None:
                writer.add_scalar("loss/val", log.val_loss, log.epoch)
            if log.lr is not None:
                writer.add_scalar("lr", log.lr, log.epoch)
            for key, value in log.extra.items():
                writer.add_scalar(key, value, log.epoch)

        if self._wandb_run:
            metrics = {
                "epoch": log.epoch,
                "train_loss": log.train_loss,
                "val_loss": log.val_loss,
                "lr": log.lr,
            }
            metrics.update(log.extra)
            self._wandb_run.log(metrics)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dict of metrics at a given step."""
        if self._tensorboard_writer is not None:
            for key, value in metrics.items():
                self._tensorboard_writer.add_scalar(key, value, step or 0)
        if self._wandb_run:
            self._wandb_run.log(metrics, step=step)

    def finish_run(self, final_metrics: dict[str, Any] | None = None) -> None:
        """Mark the end of a training run."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
        else:
            elapsed = 0.0

        summary = {
            "mode": self.mode,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "total_epochs": len(self.epoch_logs),
            "elapsed_seconds": elapsed,
        }

        if self.epoch_logs:
            best_val = min((log.val_loss for log in self.epoch_logs if log.val_loss is not None), default=None)
            summary["best_val_loss"] = best_val
            summary["final_train_loss"] = self.epoch_logs[-1].train_loss
            summary["final_val_loss"] = self.epoch_logs[-1].val_loss

        if final_metrics:
            summary["final_metrics"] = final_metrics

        summary_path = self.output_dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        epoch_logs_path = self.output_dir / "epoch_logs.json"
        epoch_logs_path.write_text(
            json.dumps([log.to_dict() for log in self.epoch_logs], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if self._tensorboard_writer is not None:
            self._tensorboard_writer.close()

        if self._wandb_run:
            self._wandb_run.finish()

    def log_figure(self, figure, name: str, step: int | None = None) -> None:
        """Log a matplotlib figure or plotly figure."""
        if self._wandb_run:
            self._wandb_run.log({name: self._wandb_run.Image(figure)}, step=step)

    @property
    def output_path(self) -> Path:
        return self.output_dir


class MetricsTracker:
    """
    Tracks and persists training metrics during an experiment.
    """

    def __init__(self):
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.extra_metrics: dict[str, list[float]] = {}
        self.best_val: float | None = None
        self.best_epoch: int = 0

    def update(
        self,
        train_loss: float,
        val_loss: float | None = None,
        extra: dict[str, float] | None = None,
        epoch: int = 0,
    ) -> None:
        """Update metrics for an epoch."""
        self.train_losses.append(float(train_loss))
        if val_loss is not None:
            self.val_losses.append(float(val_loss))
            if self.best_val is None or val_loss < self.best_val:
                self.best_val = val_loss
                self.best_epoch = epoch

        if extra:
            for key, value in extra.items():
                if key not in self.extra_metrics:
                    self.extra_metrics[key] = []
                self.extra_metrics[key].append(float(value))

    def save(self, output_dir: Path) -> None:
        """Save metrics to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.train_losses:
            np.save(output_dir / "train_losses.npy", np.array(self.train_losses))

        if self.val_losses:
            np.save(output_dir / "val_losses.npy", np.array(self.val_losses))

        for name, values in self.extra_metrics.items():
            safe_name = name.replace("/", "_").replace(" ", "_")
            np.save(output_dir / f"metric_{safe_name}.npy", np.array(values))

        summary = {
            "n_epochs": len(self.train_losses),
            "best_val": self.best_val,
            "best_epoch": self.best_epoch,
        }
        (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def get_best_state_info(self) -> dict[str, Any]:
        """Return info about the best model state."""
        return {
            "best_val": self.best_val,
            "best_epoch": self.best_epoch,
        }
