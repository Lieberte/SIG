import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GraphNorm, SAGEConv


class Chomp1d(nn.Module):
    """Remove extra elements from the end of the sequence."""

    def __init__(self, chompSize: int):
        super().__init__()
        self.chompSize = chompSize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chompSize].contiguous()


class TemporalBlock(nn.Module):
    """Dilated causal 1D convolution block for TCN."""

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
    """TCN encoder."""

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


class gcnBlock(nn.Module):
    def __init__(self, inDim: int, outDim: int):
        super().__init__()
        self.conv = GCNConv(inDim, outDim)
        self.norm = GraphNorm(outDim)

    def forward(self, x: torch.Tensor, edgeIndex: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.conv(x, edgeIndex)))


class gatBlock(nn.Module):
    def __init__(self, inDim: int, outDim: int, nHeads: int = 4, dropout: float = 0.0, edgeDim: int | None = None):
        super().__init__()
        headDim = max(1, outDim // nHeads)
        self.conv = GATConv(
            inDim, headDim, heads=nHeads,
            dropout=dropout, concat=True, edge_dim=edgeDim,
        )
        actualOut = headDim * nHeads
        self.proj = nn.Linear(actualOut, outDim) if actualOut != outDim else nn.Identity()
        self.norm = GraphNorm(outDim)

    def forward(self, x: torch.Tensor, edgeIndex: torch.Tensor, edgeAttr: torch.Tensor | None = None) -> torch.Tensor:
        out = self.conv(x, edgeIndex, edge_attr=edgeAttr)
        out = self.proj(out)
        return torch.relu(self.norm(out))


class sageBlock(nn.Module):

    def __init__(self, inDim: int, outDim: int):
        super().__init__()
        self.conv = SAGEConv(inDim, outDim)
        self.norm = GraphNorm(outDim)

    def forward(self, x: torch.Tensor, edgeIndex: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.conv(x, edgeIndex)))


def _makeConvBlock(convType: str, inDim: int, outDim: int, nHeads: int = 4, dropout: float = 0.0, edgeDim: int | None = None) -> nn.Module:
    if convType == "gat":
        return gatBlock(inDim, outDim, nHeads, dropout, edgeDim)
    elif convType == "sage":
        return sageBlock(inDim, outDim)
    else:
        return gcnBlock(inDim, outDim)


class meshGnnEncoder(nn.Module):
    def __init__(
        self,
        nNodeFeatures: int,
        hiddenSize: int,
        nLayers: int = 3,
        convType: str = "gat",
        nHeads: int = 4,
        dropout: float = 0.0,
        edgeDim: int | None = None,
    ):
        super().__init__()
        self.inputProj = nn.Linear(nNodeFeatures, hiddenSize)
        self.convType = convType
        blocks = []
        for _ in range(nLayers):
            blocks.append(_makeConvBlock(convType, hiddenSize, hiddenSize, nHeads, dropout, edgeDim))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, edgeIndex: torch.Tensor, edgeAttr: torch.Tensor | None = None) -> torch.Tensor:
        x = self.inputProj(x)
        for block in self.blocks:
            if self.convType == "gat":
                x = x + block(x, edgeIndex, edgeAttr)
            else:
                x = x + block(x, edgeIndex)
        return x


class spatialGnnModel(nn.Module):
    def __init__(
        self,
        nNodeFeatures: int,
        nOutputFeatures: int,
        hiddenSize: int = 128,
        nLayers: int = 3,
        convType: str = "gat",
        nHeads: int = 4,
        dropout: float = 0.0,
        edgeDim: int | None = None,
    ):
        super().__init__()
        self.gnnEncoder = meshGnnEncoder(nNodeFeatures, hiddenSize, nLayers, convType, nHeads, dropout, edgeDim)
        self.decoder = nn.Linear(hiddenSize, nOutputFeatures)

    def forward(self, x: torch.Tensor, edgeIndex: torch.Tensor, edgeAttr: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.gnnEncoder(x, edgeIndex, edgeAttr)
        return self.decoder(encoded)


class temporalMeshGnn(nn.Module):
    """
    Temporal mesh GNN: spatial GNN encoding + temporal modeling.

    Supports two temporal encoders:
    - 'gru': GRU (default, good for short sequences)
    - 'tcn': Temporal Convolutional Network (better for capturing local patterns)
    """

    def __init__(
        self,
        nNodeFeatures: int,
        nOutputFeatures: int,
        hiddenSize: int = 128,
        nConvLayers: int = 3,
        convType: str = "gat",
        nHeads: int = 4,
        nGruLayers: int = 1,
        dropout: float = 0.0,
        edgeDim: int | None = None,
        temporalType: str = "gru",
        kernelSize: int = 3,
    ):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.temporalType = temporalType

        self.gnnEncoder = meshGnnEncoder(
            nNodeFeatures, hiddenSize, nConvLayers, convType, nHeads, dropout, edgeDim,
        )

        if temporalType == "tcn":
            numChannels = [hiddenSize] * nGruLayers
            self.temporal = TemporalConvNet(hiddenSize, numChannels, kernelSize, dropout)
        else:
            self.temporal = nn.GRU(
                hiddenSize, hiddenSize,
                num_layers=nGruLayers, batch_first=True,
                dropout=dropout if nGruLayers > 1 else 0.0,
            )

        self.decoder = nn.Sequential(
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, nOutputFeatures),
        )
        self.convType = convType

    def _encodeFrame(
        self, nodeFeatures: torch.Tensor, edgeIndex: torch.Tensor, edgeAttr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.gnnEncoder(nodeFeatures, edgeIndex, edgeAttr)

    def _encodeBatchFrame(
        self,
        nodeFeatures: torch.Tensor,
        edgeIndex: torch.Tensor,
        edgeAttr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, nNodes, nFeat = nodeFeatures.shape
        # Offset edge indices for each graph in the batch
        offsets = torch.arange(batch, device=nodeFeatures.device).unsqueeze(1) * nNodes
        batchEdgeIndex = edgeIndex.unsqueeze(0) + offsets  # (batch, 2, nEdges)
        batchEdgeIndex = batchEdgeIndex.permute(1, 0, 2).reshape(2, -1)  # (2, batch*nEdges)

        if edgeAttr is not None:
            batchEdgeAttr = edgeAttr.repeat(batch, 1)  # (batch*nEdges, edgeDim)
        else:
            batchEdgeAttr = None

        # Flatten node features: (batch*nNodes, feat)
        flatNodes = nodeFeatures.reshape(batch * nNodes, nFeat)
        flatEncoded = self.gnnEncoder(flatNodes, batchEdgeIndex, batchEdgeAttr)
        return flatEncoded.reshape(batch, nNodes, self.hiddenSize)

    def forward(
        self,
        nodeFeatureSeq: torch.Tensor,
        edgeIndex: torch.Tensor,
        edgeAttr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seqLen, nNodes, nFeat = nodeFeatureSeq.shape

        # Encode each time step with batched graph processing
        allEmbeddings = []
        for t in range(seqLen):
            emb = self._encodeBatchFrame(nodeFeatureSeq[:, t], edgeIndex, edgeAttr)
            allEmbeddings.append(emb)

        # (batch, seqLen, nNodes, hiddenSize)
        embeddings = torch.stack(allEmbeddings, dim=1)

        # Temporal modeling
        if self.temporalType == "tcn":
            # TCN expects (batch*nNodes, hiddenSize, seqLen)
            flat = embeddings.permute(0, 2, 3, 1).reshape(
                batch * nNodes, self.hiddenSize, seqLen
            )
            temporal = self.temporal(flat)
            # Take last time step: (batch*nNodes, hiddenSize, 1)
            last = temporal[:, :, -1].reshape(batch, nNodes, -1)
        else:
            # GRU expects (batch*nNodes, seqLen, hiddenSize)
            flat = embeddings.permute(0, 2, 1, 3).reshape(batch * nNodes, seqLen, self.hiddenSize)
            temporal, _ = self.temporal(flat)
            last = temporal[:, -1, :].reshape(batch, nNodes, -1)

        return self.decoder(last)

    @torch.no_grad()
    def rollout(
        self,
        initSeq: torch.Tensor,
        edgeIndex: torch.Tensor,
        nSteps: int,
        edgeAttr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        window = initSeq.to(device)
        preds = []
        for _ in range(nSteps):
            pred = self.forward(window, edgeIndex, edgeAttr)
            preds.append(pred.squeeze(0))
            window = torch.cat([window[:, 1:, :, :], pred.unsqueeze(1)], dim=1)
        return torch.stack(preds)
