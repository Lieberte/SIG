import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class romTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lossFn: nn.Module,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lossFn = lossFn
        self.device = device
        self.history: dict[str, list[float]] = {"trainLoss": [], "valLoss": []}

    def _extractPrediction(self, output: torch.Tensor | tuple) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output

    def trainEpoch(self, loader: DataLoader) -> float:
        self.model.train()
        totalLoss = 0.0
        count = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self._extractPrediction(self.model(x))
            loss = self.lossFn(pred, y)
            loss.backward()
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
            pred = self._extractPrediction(self.model(x))
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
                    bestState = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    noImproveCount += 1
            if verbose:
                msg = f"epoch {epoch + 1}/{nEpochs}  trainLoss={trainLoss:.6f}"
                if valLoss is not None:
                    msg += f"  valLoss={valLoss:.6f}"
                print(msg)
            if patience > 0 and noImproveCount >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch + 1}")
                break
        if bestState is not None:
            self.model.load_state_dict(bestState)
        return self.history
