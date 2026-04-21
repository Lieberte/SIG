from typing import Any

import torch
import torch.nn as nn

from training.pinn.physicsLoss import (
    continuityLoss,
    energyLoss,
    fieldGradients,
    momentumXLoss,
    momentumYLoss,
    vofLoss,
)


class residualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class pinnModel(nn.Module):
    def __init__(
        self,
        nSpatialDim: int = 3,
        nBcParams: int = 0,
        nOutputFields: int = 6,
        hiddenSize: int = 128,
        nBlocks: int = 4,
    ):
        super().__init__()
        self.nSpatialDim = nSpatialDim
        self.nBcParams = nBcParams
        self.nOutputFields = nOutputFields
        inputDim = nSpatialDim + 1 + nBcParams
        self.inputProj = nn.Sequential(nn.Linear(inputDim, hiddenSize), nn.Tanh())
        self.blocks = nn.Sequential(*[residualBlock(hiddenSize) for _ in range(nBlocks)])
        self.outputProj = nn.Linear(hiddenSize, nOutputFields)
        self.fieldNames: list[str] = []

    def forward(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
        bcParams: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [coords, t]
        if bcParams is not None:
            parts.append(bcParams)
        x = torch.cat(parts, dim=-1)
        x = self.inputProj(x)
        x = self.blocks(x)
        return self.outputProj(x)

    def predictFields(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
        bcParams: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raw = self.forward(coords, t, bcParams)
        if not self.fieldNames:
            return {f"field_{i}": raw[:, i : i + 1] for i in range(raw.shape[1])}
        return {name: raw[:, i : i + 1] for i, name in enumerate(self.fieldNames)}


class pinnLossComposer:
    def __init__(
        self,
        model: pinnModel,
        physicsParams: dict[str, Any],
        lambdaData: float = 1.0,
        lambdaPhysics: float = 1.0,
        lambdaBc: float = 1.0,
    ):
        self.model = model
        self.mu = physicsParams.get("mu", 1e-3)
        self.cp = physicsParams.get("cp", 1000.0)
        self.k = physicsParams.get("k", 0.6)
        self.lambdaData = lambdaData
        self.lambdaPhysics = lambdaPhysics
        self.lambdaBc = lambdaBc
        self.continuity = continuityLoss()
        self.momX = momentumXLoss()
        self.momY = momentumYLoss()
        self.energy = energyLoss()
        self.vof = vofLoss()
        self.enabledPdes: set[str] = {"continuity", "momentumX", "momentumY", "energy", "vof"}

    def dataLoss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return nn.functional.mse_loss(prediction, target)

    def physicsLoss(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
        bcParams: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        coords = coords.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        fields = self.model.predictFields(coords, t, bcParams)
        u = fields.get("u", fields.get("field_0", torch.zeros(coords.shape[0], 1)))
        v = fields.get("v", fields.get("field_1", torch.zeros(coords.shape[0], 1)))
        w = fields.get("w") or fields.get("field_2")
        p = fields.get("p") or fields.get("field_3")
        T = fields.get("T") or fields.get("field_4")
        rho = fields.get("rho") or fields.get("field_5")
        alpha = fields.get("alpha")
        if rho is None:
            rho = torch.ones(coords.shape[0], 1, device=coords.device)
        allFieldsForGrad = {"u": u, "v": v, "rho": rho}
        if w is not None:
            allFieldsForGrad["w"] = w
        if p is not None:
            allFieldsForGrad["p"] = p
        if T is not None:
            allFieldsForGrad["T"] = T
        if alpha is not None:
            allFieldsForGrad["alpha"] = alpha
        grads = fieldGradients(allFieldsForGrad, coords, t)
        losses: dict[str, torch.Tensor] = {}
        if "continuity" in self.enabledPdes:
            losses["continuity"] = self.continuity(rho, u, v, w, grads)
        if "momentumX" in self.enabledPdes:
            losses["momentumX"] = self.momX(rho, u, v, w, self.mu, grads, coords)
        if "momentumY" in self.enabledPdes:
            losses["momentumY"] = self.momY(rho, u, v, w, self.mu, grads, coords)
        if "energy" in self.enabledPdes and T is not None:
            losses["energy"] = self.energy(rho, self.cp, self.k, T, u, v, w, grads, coords)
        if "vof" in self.enabledPdes and alpha is not None:
            losses["vof"] = self.vof(alpha, u, v, w, grads)
        return losses

    def compositeLoss(
        self,
        coordsData: torch.Tensor,
        tData: torch.Tensor,
        predData: torch.Tensor,
        targetData: torch.Tensor,
        coordsColloc: torch.Tensor,
        tColloc: torch.Tensor,
        bcParamsColloc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        dLoss = self.dataLoss(predData, targetData)
        pLosses = self.physicsLoss(coordsColloc, tColloc, bcParamsColloc)
        pTotal = sum(pLosses.values()) if pLosses else torch.tensor(0.0)
        total = self.lambdaData * dLoss + self.lambdaPhysics * pTotal
        breakdown = {"data": dLoss.item(), "physics": pTotal.item() if isinstance(pTotal, torch.Tensor) else 0.0}
        for k, v in pLosses.items():
            breakdown[k] = v.item()
        return total, breakdown
