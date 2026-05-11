import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Remove extra elements from the end of the sequence."""

    def __init__(self, chompSize: int):
        super().__init__()
        self.chompSize = chompSize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chompSize].contiguous()


class TemporalBlock(nn.Module):
    """Dilated causal 1D convolution block (TCN building block)."""

    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernelSize - 1) * dilation
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(inChannels, outChannels, kernelSize, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(outChannels, outChannels, kernelSize, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(inChannels, outChannels, 1) if inChannels != outChannels else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network encoder."""

    def __init__(
        self,
        numInputs: int,
        numChannels: list[int],
        kernelSize: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        for i, nOut in enumerate(numChannels):
            inCh = numInputs if i == 0 else numChannels[i - 1]
            dilation = 2 ** i
            layers.append(TemporalBlock(inCh, nOut, kernelSize, dilation, dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class bcToSurfaceModel(nn.Module):
    """
    End-to-end model: boundary conditions → surface field prediction.

    Supports two encoder types:
    - 'tcn': Temporal Convolutional Network (recommended for short sequences)
    - 'lstm': LSTM (for longer sequences)
    """

    def __init__(
        self,
        nInputFeatures: int,
        nOutputFeatures: int,
        hiddenSize: int = 128,
        nLayers: int = 2,
        dropout: float = 0.1,
        encoderType: str = "tcn",
        kernelSize: int = 3,
    ):
        super().__init__()
        self.encoderType = encoderType
        self.nOutputFeatures = nOutputFeatures

        if encoderType == "tcn":
            numChannels = [hiddenSize] * nLayers
            self.encoder = TemporalConvNet(
                nInputFeatures, numChannels, kernelSize, dropout,
            )
            self.decoder = nn.Sequential(
                nn.Linear(hiddenSize, hiddenSize),
                nn.ReLU(),
                nn.Linear(hiddenSize, nOutputFeatures),
            )
        else:
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
        """
        Args:
            x: (batch, seqLen, nInputFeatures)
        Returns:
            (batch, nOutputFeatures)
        """
        if self.encoderType == "tcn":
            # TCN expects (batch, channels, seqLen)
            x = x.transpose(1, 2)
            encoded = self.encoder(x)
            # Take last time step: (batch, hiddenSize, 1) → (batch, hiddenSize)
            h = encoded[:, :, -1]
        else:
            _, (h, _) = self.encoder(x)
            h = h[-1]
        return self.decoder(h)

    @torch.no_grad()
    def rollout(
        self,
        bcHistory: torch.Tensor,
        bcFuture: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive rollout for multi-step prediction."""
        self.eval()
        device = next(self.parameters()).device
        window = bcHistory.to(device)
        preds = []
        for t in range(bcFuture.shape[0]):
            pred = self.forward(window)
            preds.append(pred.squeeze(0))
            newBc = bcFuture[t : t + 1].to(device).unsqueeze(0)
            window = torch.cat([window[:, 1:, :], newBc], dim=1)
        return torch.stack(preds)


class rnnStateUpdater(nn.Module):
    """
    Hand-written recurrent state updater (Elman-style RNN).

    Explicit state update equation at each time step:

        h_t = act(W_hh @ h_{t-1} + W_xh @ x_t + b_h)   # state transition
        y_t = W_hy @ h_t + b_y                           # output projection

    Compared to PyTorch nn.RNN:
    - Fully transparent: every matrix and bias is a named parameter
    - Easy to inspect weights, add constraints, or modify dynamics
    - Single-layer only (no stacking, no bidirectional)
    """

    def __init__(
        self,
        nInputFeatures: int,
        hiddenSize: int,
        nOutputFeatures: int,
        activation: str = "tanh",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.act_fn = {
            "tanh": torch.tanh,
            "relu": torch.relu,
        }.get(activation.lower(), torch.tanh)

        # State transition: h_t = act(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
        self.W_hh = nn.Parameter(torch.randn(hiddenSize, hiddenSize) * 0.1)
        self.W_xh = nn.Parameter(torch.randn(hiddenSize, nInputFeatures) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hiddenSize))

        # Output projection: y_t = W_hy @ h_t + b_y
        self.W_hy = nn.Parameter(torch.randn(nOutputFeatures, hiddenSize) * 0.1)
        self.b_y = nn.Parameter(torch.zeros(nOutputFeatures))

        self.dropout = nn.Dropout(dropout)

    def reset_state(self, batch: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state to zero."""
        return torch.zeros(batch, self.hiddenSize, device=device)

    def step(self, x: torch.Tensor, h_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single time step.

        Args:
            x: (batch, nInputFeatures) — current input
            h_prev: (batch, hiddenSize) — previous hidden state

        Returns:
            y: (batch, nOutputFeatures) — current output
            h_new: (batch, hiddenSize) — updated hidden state
        """
        # h_t = act(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
        h_new = self.act_fn(
            h_prev @ self.W_hh.T + x @ self.W_xh.T + self.b_h
        )
        h_new = self.dropout(h_new)
        # y_t = W_hy @ h_t + b_y
        y = h_new @ self.W_hy.T + self.b_y
        return y, h_new

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full sequence forward pass.

        Args:
            x: (batch, seqLen, nInputFeatures)

        Returns:
            outputs: (batch, seqLen, nOutputFeatures) — output at every step
            h_final: (batch, hiddenSize) — final hidden state
        """
        batch, seqLen, _ = x.shape
        device = x.device
        h = self.reset_state(batch, device)
        outputs = []
        for t in range(seqLen):
            y, h = self.step(x[:, t], h)
            outputs.append(y)
        outputs = torch.stack(outputs, dim=1)  # (batch, seqLen, nOutputFeatures)
        return outputs, h

    @torch.no_grad()
    def rollout(
        self,
        bcHistory: torch.Tensor,
        bcFuture: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive multi-step prediction."""
        self.eval()
        device = next(self.parameters()).device
        batch, seqLen, _ = bcHistory.shape

        # Warm-up: encode history
        h = self.reset_state(batch, device)
        for t in range(seqLen):
            _, h = self.step(bcHistory[:, t].to(device), h)

        # Predict future steps
        preds = []
        for t in range(bcFuture.shape[0]):
            y, h = self.step(bcFuture[t : t + 1].to(device), h)
            preds.append(y.squeeze(0))
        return torch.stack(preds)


class rnnModel(nn.Module):
    """
    Wrapper around rnnStateUpdater for use in the training pipeline.

    Returns the final-step output (matching bcToSurfaceModel interface).
    For per-step outputs, use rnnStateUpdater directly.
    """

    def __init__(
        self,
        nInputFeatures: int,
        nOutputFeatures: int,
        hiddenSize: int = 128,
        activation: str = "tanh",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.core = rnnStateUpdater(
            nInputFeatures, hiddenSize, nOutputFeatures, activation, dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seqLen, nInputFeatures)
        Returns:
            (batch, nOutputFeatures) — final-step prediction
        """
        outputs, _ = self.core(x)
        return outputs[:, -1, :]

    @torch.no_grad()
    def rollout(self, bcHistory: torch.Tensor, bcFuture: torch.Tensor) -> torch.Tensor:
        return self.core.rollout(bcHistory, bcFuture)


class mlpModel(nn.Module):
    """
    Simple MLP model for one-to-one prediction (no time window).

    Each time step is predicted independently from its boundary conditions.
    Best for very short time series or when temporal dependency is weak.
    """

    def __init__(
        self,
        nInputFeatures: int,
        nOutputFeatures: int,
        hiddenSize: int = 128,
        nHiddenLayers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(
            activation.lower(), nn.ReLU
        )
        layers = []
        inDim = nInputFeatures
        for _ in range(nHiddenLayers - 1):
            layers.extend([
                nn.Linear(inDim, hiddenSize),
                act_fn(),
                nn.Dropout(dropout),
            ])
            inDim = hiddenSize
        layers.append(nn.Linear(inDim, nOutputFeatures))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, nInputFeatures) or (batch, seqLen, nInputFeatures)
        Returns:
            (batch, nOutputFeatures)
        """
        if x.dim() == 3:
            x = x[:, -1, :]  # Take last time step for compatibility
        return self.net(x)
