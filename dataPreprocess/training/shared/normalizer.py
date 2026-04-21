import numpy as np


class standardScaler:
    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "standardScaler":
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        self.std = np.where(self.std < 1e-12, 1.0, self.std)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverseTransform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def fitTransform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def saveParams(self) -> dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    def loadParams(self, params: dict[str, np.ndarray]) -> "standardScaler":
        self.mean = params["mean"]
        self.std = params["std"]
        return self
