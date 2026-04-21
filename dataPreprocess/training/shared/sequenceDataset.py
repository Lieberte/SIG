import numpy as np
import torch
from torch.utils.data import Dataset


class timeSeriesDataset(Dataset):
    def __init__(self, inputSeq: np.ndarray, targetSeq: np.ndarray, seqLen: int = 10):
        self.inputSeq = torch.from_numpy(np.asarray(inputSeq, dtype=np.float32))
        self.targetSeq = torch.from_numpy(np.asarray(targetSeq, dtype=np.float32))
        self.seqLen = seqLen
        self.nSamples = max(0, inputSeq.shape[0] - seqLen)

    def __len__(self) -> int:
        return self.nSamples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.inputSeq[idx : idx + self.seqLen]
        y = self.targetSeq[idx + self.seqLen]
        return x, y
