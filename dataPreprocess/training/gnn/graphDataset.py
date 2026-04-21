import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class meshGraphDataset(Dataset):
    def __init__(
        self,
        nodeFeatures: np.ndarray,
        edgeIndex: np.ndarray,
        seqLen: int = 10,
        edgeAttr: np.ndarray | None = None,
    ):
        self.nodeFeatures = torch.from_numpy(nodeFeatures.astype(np.float32))
        self.edgeIndex = torch.from_numpy(edgeIndex.astype(np.int64))
        self.edgeAttr = (
            torch.from_numpy(edgeAttr.astype(np.float32))
            if edgeAttr is not None
            else None
        )
        self.seqLen = seqLen
        self.nSamples = max(0, nodeFeatures.shape[0] - seqLen)

    def __len__(self) -> int:
        return self.nSamples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        xFrames = self.nodeFeatures[idx : idx + self.seqLen]
        yFrame = self.nodeFeatures[idx + self.seqLen]
        return xFrames, yFrame

    def getEdgeIndex(self) -> torch.Tensor:
        return self.edgeIndex

    def getEdgeAttr(self) -> torch.Tensor | None:
        return self.edgeAttr

    def toPygData(self, idx: int) -> Data:
        xFrames, yFrame = self[idx]
        return Data(
            x=xFrames[-1],
            edge_index=self.edgeIndex,
            edge_attr=self.edgeAttr,
            y=yFrame,
        )

    def toPygSequence(self, idx: int) -> list[Data]:
        xFrames, yFrame = self[idx]
        sequence = []
        for t in range(xFrames.shape[0]):
            sequence.append(Data(
                x=xFrames[t],
                edge_index=self.edgeIndex,
                edge_attr=self.edgeAttr,
            ))
        return sequence
