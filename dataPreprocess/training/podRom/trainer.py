"""
POD-ROM specific trainer with rollout evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.shared.dataSplit import temporal_split
from training.shared.evaluationMetrics import compute_metrics, compute_rollout_error, evaluate_predictions
from training.shared.normalizer import standardScaler
from training.shared.podBasis import podBasis
from training.shared.sequenceDataset import timeSeriesDataset
from training.shared.trainingLogger import ExperimentLogger, RunConfig, EpochLog, MetricsTracker


class PodRomTrainer:
    """Trainer for POD-ROM models with rollout evaluation."""

    def __init__(
        self,
        model: nn.Module,
        pod: podBasis,
        device: str = "cpu",
        learningRate: float = 1e-3,
        weightDecay: float = 0.0,
    ):
        self.model = model.to(device)
        self.pod = pod
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learningRate, weight_decay=weightDecay
        )
        self.scheduler = None
        self.lossFn = nn.MSELoss()
        self.history: dict[str, list[float]] = {"trainLoss": [], "valLoss": []}

    def setScheduler(self, scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:
        self.scheduler = scheduler

    def _makeLoader(
        self,
        latent: np.ndarray,
        bc: np.ndarray,
        seqLen: int,
        batchSize: int,
        shuffle: bool = True,
    ) -> DataLoader:
        # Combine latent and BC as input
        combined = np.concatenate([latent, bc], axis=1)
        dataset = timeSeriesDataset(combined, latent, seqLen=seqLen)
        return DataLoader(dataset, batch_size=batchSize, shuffle=shuffle)

    def trainEpoch(self, loader: DataLoader) -> float:
        self.model.train()
        totalLoss = 0.0
        count = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.lossFn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            totalLoss += loss.item() * x.size(0)
            count += x.size(0)
        return totalLoss / max(count, 1)

    @torch.no_grad()
    def evalEpoch(self, loader: DataLoader) -> float:
        self.model.eval()
        totalLoss = 0.0
        count = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss = self.lossFn(pred, y)
            totalLoss += loss.item() * x.size(0)
            count += x.size(0)
        return totalLoss / max(count, 1)

    def fit(
        self,
        trainLoader: DataLoader,
        valLoader: DataLoader | None = None,
        nEpochs: int = 100,
        patience: int = 10,
        verbose: bool = True,
        logger: ExperimentLogger | None = None,
    ) -> dict[str, list[float]]:
        bestValLoss = float("inf")
        noImproveCount = 0
        bestState: dict | None = None

        for epoch in range(nEpochs):
            trainLoss = self.trainEpoch(trainLoader)
            self.history["trainLoss"].append(trainLoss)

            valLoss: float | None = None
            if valLoader is not None:
                valLoss = self.evalEpoch(valLoader)
                self.history["valLoss"].append(valLoss)

                if valLoss < bestValLoss:
                    bestValLoss = valLoss
                    noImproveCount = 0
                    bestState = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    noImproveCount += 1

            if self.scheduler is not None:
                self.scheduler.step()

            if logger is not None:
                logger.log_epoch(EpochLog(
                    epoch=epoch,
                    train_loss=trainLoss,
                    val_loss=valLoss,
                ))

            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch}/{nEpochs}  trainLoss={trainLoss:.6f}"
                if valLoss is not None:
                    msg += f"  valLoss={valLoss:.6f}"
                print(msg)

            if patience > 0 and noImproveCount >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        if bestState is not None:
            self.model.load_state_dict(bestState)

        return self.history

    @torch.no_grad()
    def evaluateRollout(
        self,
        latent: np.ndarray,
        bc: np.ndarray,
        seqLen: int,
        nSteps: int,
    ) -> dict[str, Any]:
        """Evaluate autoregressive rollout performance."""
        self.model.eval()

        # Use first seqLen steps as initial window
        window = torch.from_numpy(
            np.concatenate([latent[:seqLen], bc[:seqLen]], axis=1)
        ).float().unsqueeze(0).to(self.device)

        predictions = []
        targets = []

        for t in range(seqLen, min(seqLen + nSteps, len(latent))):
            pred = self.model(window)
            predictions.append(pred.squeeze(0).cpu().numpy())
            targets.append(latent[t])

            # Update window
            bc_t = torch.from_numpy(bc[t]).float().unsqueeze(0).to(self.device)
            newStep = torch.cat([pred, bc_t], dim=1).unsqueeze(1)
            window = torch.cat([window[:, 1:, :], newStep], dim=1)

        if not predictions:
            return {"rollout_rmse": float("nan"), "n_steps": 0}

        predictions = np.array(predictions)
        targets = np.array(targets)

        metrics = compute_metrics(predictions, targets)
        return {
            "rollout_rmse": metrics.rmse,
            "rollout_mae": metrics.mae,
            "rollout_r2": metrics.r2,
            "n_steps": len(predictions),
        }

    @torch.no_grad()
    def reconstructField(
        self,
        latentCoefficients: np.ndarray,
        scaler: standardScaler | None = None,
    ) -> np.ndarray:
        """Reconstruct full field from latent coefficients."""
        if scaler is not None:
            latentCoefficients = scaler.inverseTransform(latentCoefficients)

        # Decode using POD
        return self.pod.decode(latentCoefficients.T)

    def save(self, outputDir: Path) -> None:
        """Save model and training state."""
        outputDir = Path(outputDir)
        outputDir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), outputDir / "model.pt")
        self.pod.save(outputDir / "pod")

        np.save(outputDir / "trainLoss.npy", np.array(self.history["trainLoss"]))
        if self.history["valLoss"]:
            np.save(outputDir / "valLoss.npy", np.array(self.history["valLoss"]))

        report = {
            "finalTrainLoss": self.history["trainLoss"][-1] if self.history["trainLoss"] else None,
            "finalValLoss": self.history["valLoss"][-1] if self.history["valLoss"] else None,
            "nEpochs": len(self.history["trainLoss"]),
        }
        (outputDir / "trainReport.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
        )
