import torch
import torch.nn as nn


class bcToSurfaceModel(nn.Module):
    def __init__(
        self,
        nInputFeatures: int,
        nOutputFeatures: int,
        hiddenSize: int = 128,
        nLayers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.nInputFeatures = nInputFeatures
        self.nOutputFeatures = nOutputFeatures
        self.encoder = nn.LSTM(
            nInputFeatures,
            hiddenSize,
            num_layers=nLayers,
            batch_first=True,
            dropout=dropout if nLayers > 1 else 0.0,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, nOutputFeatures),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        return self.decoder(h[-1])

    @torch.no_grad()
    def rollout(
        self,
        bcHistory: torch.Tensor,
        bcFuture: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        window = bcHistory.to(device)
        preds: list[torch.Tensor] = []
        for t in range(bcFuture.shape[0]):
            pred = self.forward(window)
            preds.append(pred.squeeze(0))
            newBc = bcFuture[t : t + 1].to(device).unsqueeze(0)
            window = torch.cat([window[:, 1:, :], newBc], dim=1)
        return torch.stack(preds)
