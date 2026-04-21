import torch
import torch.nn as nn


class latentDynamicsModel(nn.Module):
    """Latent dynamics model for POD-ROM."""

    def __init__(
        self,
        nLatent: int,
        nBcFeatures: int,
        hiddenSize: int = 128,
        nLayers: int = 2,
        dropout: float = 0.1,
        activation: str = "tanh",
    ):
        super().__init__()
        self.nLatent = nLatent
        self.nBcFeatures = nBcFeatures
        activation_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }.get(activation.lower(), nn.Tanh)

        layers = []
        in_dim = nLatent + nBcFeatures
        for _ in range(nLayers - 1):
            layers.extend([
                nn.Linear(in_dim, hiddenSize),
                activation_fn(),
                nn.Dropout(dropout),
            ])
            in_dim = hiddenSize
        layers.append(nn.Linear(in_dim, nLatent))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def rollout(
        self,
        latentBcHistory: torch.Tensor,
        bcFuture: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        window = latentBcHistory
        preds = []
        for t in range(bcFuture.shape[0]):
            pred = self.forward(window)
            preds.append(pred.squeeze(0))
            bc = bcFuture[t : t + 1]
            newStep = torch.cat([pred, bc], dim=1).unsqueeze(1)
            window = torch.cat([window[:, 1:, :], newStep], dim=1)
        return torch.stack(preds)


class podRomSurfaceModel(nn.Module):
    """POD-ROM surface model: dynamics + surface reconstruction."""

    def __init__(
        self,
        nLatent: int,
        nBcFeatures: int,
        nSurfaceCells: int,
        hiddenSize: int = 128,
        nLayers: int = 2,
        dropout: float = 0.1,
        activation: str = "tanh",
    ):
        super().__init__()
        self.dynamics = latentDynamicsModel(
            nLatent, nBcFeatures, hiddenSize, nLayers, dropout, activation,
        )
        self.surfaceHead = nn.Sequential(
            nn.Linear(nLatent, hiddenSize),
            self._make_activation(activation),
            nn.Linear(hiddenSize, nSurfaceCells),
        )

    @staticmethod
    def _make_activation(activation: str) -> nn.Module:
        return {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }.get(activation.lower(), nn.Tanh)()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latentPred = self.dynamics(x)
        surfacePred = self.surfaceHead(latentPred)
        return latentPred, surfacePred

    @torch.no_grad()
    def rollout(
        self,
        latentBcHistory: torch.Tensor,
        bcFuture: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        latentPreds = self.dynamics.rollout(latentBcHistory, bcFuture)
        surfacePreds = self.surfaceHead(latentPreds)
        return latentPreds, surfacePreds