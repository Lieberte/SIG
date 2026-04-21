import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GraphNorm, SAGEConv, MessagePassing


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
    Temporal mesh GNN model.

    Uses efficient batch encoding by reshaping (batch*seqLen, nNodes, feat)
    into a single forward pass through the GNN encoder.
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
    ):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.gnnEncoder = meshGnnEncoder(
            nNodeFeatures, hiddenSize, nConvLayers, convType, nHeads, dropout, edgeDim,
        )
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
        """
        Encode a batch of frames efficiently using PyG-style batching.

        Args:
            nodeFeatures: (batch, nNodes, feat)
            edgeIndex: (2, nEdges) - shared topology
            edgeAttr: (nEdges, edgeDim) or None

        Returns:
            (batch, nNodes, hiddenSize)
        """
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
            # (batch, nNodes, hiddenSize)
            emb = self._encodeBatchFrame(nodeFeatureSeq[:, t], edgeIndex, edgeAttr)
            allEmbeddings.append(emb)

        # (batch, seqLen, nNodes, hiddenSize)
        embeddings = torch.stack(allEmbeddings, dim=1)

        # Temporal: permute to (batch*nNodes, seqLen, hiddenSize) for GRU
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
