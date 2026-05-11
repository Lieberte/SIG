from typing import Any

import numpy as np
import torch
import torch.nn as nn

from training.pinn.physicsLoss import (
    continuityLoss,
    energyLoss,
    fieldGradients,
    inletBCLoss,
    momentumXLoss,
    momentumYLoss,
    solidEnergyLoss,
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


# ============================================================
# 参数化PINN训练器（支持多工况：不同预热/双氧水温度组合）
# ============================================================
class ParametricPINNTrainer:
    """
    支持参数化入口温度的PINN训练器
    使用方法:
        trainer = ParametricPINNTrainer(model, inlet_a_coords, inlet_b_coords)
        for epoch in range(n_epochs):
            T_preheat = torch.tensor([[160.0 + 273.15]])  # 160°C
            T_h2o2 = torch.tensor([[200.0 + 273.15]])     # 200°C
            loss = trainer.train_step(coords, t, target, T_preheat, T_h2o2)
    """
    def __init__(
        self,
        model: pinnModel,
        inlet_a_coords: np.ndarray,
        inlet_b_coords: np.ndarray,
        lr: float = 1e-4,
    ):
        self.model = model
        device = next(model.parameters()).device

        # 入口坐标
        self.inlet_a = torch.tensor(inlet_a_coords, dtype=torch.float32, device=device)
        self.inlet_b = torch.tensor(inlet_b_coords, dtype=torch.float32, device=device)

        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
        target: torch.Tensor,
        T_preheat: torch.Tensor,  # [1, 1] 预热温度参数
        T_h2o2: torch.Tensor,     # [1, 1] 双氧水温度参数
        bc_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """单步训练"""
        self.optimizer.zero_grad()

        # 扩展bcParams到所有点
        bc_params = torch.cat([T_preheat, T_h2o2], dim=-1).expand(len(coords), 2)

        # 预测
        pred = self.model(coords, t, bc_params)

        # 数据损失
        loss_data = nn.functional.mse_loss(pred, target)

        # 入口边界条件损失（直接在入口面强制UDF时序温度）
        t_current = t.mean().reshape(1, 1)
        bc_params_a = torch.cat([T_preheat, T_h2o2], dim=-1).expand(len(self.inlet_a), 2)
        bc_params_b = torch.cat([T_preheat, T_h2o2], dim=-1).expand(len(self.inlet_b), 2)

        t_a = t_current.expand(len(self.inlet_a), 1)
        t_b = t_current.expand(len(self.inlet_b), 1)

        pred_inlet_a = self.model.predictFields(self.inlet_a, t_a, bc_params_a)
        pred_inlet_b = self.model.predictFields(self.inlet_b, t_b, bc_params_b)

        T_pred_a = pred_inlet_a.get("T") or pred_inlet_a.get("field_4")
        T_pred_b = pred_inlet_b.get("T") or pred_inlet_b.get("field_4")

        # 使用UDF时序计算目标温度
        from training.pinn.physicsLoss import udf_inlet_temperature
        T_target_a = udf_inlet_temperature(t_a, 'inlet_a')
        T_target_b = udf_inlet_temperature(t_b, 'inlet_b')

        loss_bc = nn.functional.mse_loss(T_pred_a, T_target_a) + \
                  nn.functional.mse_loss(T_pred_b, T_target_b)

        # 总损失
        total_loss = 1.0 * loss_data + 10.0 * loss_bc

        total_loss.backward()
        self.optimizer.step()

        return total_loss, {
            'loss_data': loss_data.item(),
            'loss_bc': loss_bc.item(),
            'T_preheat': T_preheat.item(),
            'T_h2o2': T_h2o2.item(),
        }

    def predict(
        self,
        coords: np.ndarray,
        t: float,
        T_preheat: float,
        T_h2o2: float,
    ) -> dict[str, np.ndarray]:
        """预测：给定坐标、时间、入口温度参数，返回流场"""
        device = next(self.model.parameters()).device
        coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
        t_t = torch.tensor([[t]], dtype=torch.float32, device=device).expand(len(coords), 1)
        T_preheat_t = torch.tensor([[T_preheat]], dtype=torch.float32, device=device)
        T_h2o2_t = torch.tensor([[T_h2o2]], dtype=torch.float32, device=device)

        bc_params = torch.cat([T_preheat_t, T_h2o2_t], dim=-1).expand(len(coords), 2)

        self.model.eval()
        with torch.no_grad():
            fields = self.model.predictFields(coords_t, t_t, bc_params)
        self.model.train()

        return {k: v.cpu().numpy() for k, v in fields.items()}


class pinnLossComposer:
    def __init__(
        self,
        model: pinnModel,
        physicsParams: dict[str, Any],
        inletACoords: np.ndarray | None = None,
        inletBCoords: np.ndarray | None = None,
        solidMask: np.ndarray | None = None,
        lambdaData: float = 1.0,
        lambdaPhysics: float = 1.0,
        lambdaBc: float = 10.0,  # 边界条件权重更大
        lambdaSolid: float = 5.0,  # 固体温度权重更大
    ):
        self.model = model
        self.mu = physicsParams.get("mu", 1e-3)
        self.cp_fluid = physicsParams.get("cp_fluid", 1000.0)
        self.k_fluid = physicsParams.get("k_fluid", 0.6)
        self.rho_fluid = physicsParams.get("rho_fluid", 1.2)

        # 固体物性（铝 - 你的案例是soild:1）
        self.rho_solid = physicsParams.get("rho_solid", 2700.0)
        self.cp_solid = physicsParams.get("cp_solid", 900.0)
        self.k_solid = physicsParams.get("k_solid", 200.0)

        # 损失权重
        self.lambdaData = lambdaData
        self.lambdaPhysics = lambdaPhysics
        self.lambdaBc = lambdaBc
        self.lambdaSolid = lambdaSolid

        # PDE损失
        self.continuity = continuityLoss()
        self.momX = momentumXLoss()
        self.momY = momentumYLoss()
        self.energy = energyLoss()
        self.vof = vofLoss()
        self.solidEnergy = solidEnergyLoss()

        # 边界条件损失
        self.inletBC: inletBCLoss | None = None
        if inletACoords is not None and inletBCoords is not None:
            device = next(model.parameters()).device
            inlet_a = torch.tensor(inletACoords, dtype=torch.float32, device=device)
            inlet_b = torch.tensor(inletBCoords, dtype=torch.float32, device=device)
            self.inletBC = inletBCLoss(inlet_a, inlet_b, use_parametric=model.nBcParams >= 2)

        # 固体区域掩码
        self.solidMask = solidMask

        self.enabledPdes: set[str] = {
            "continuity", "momentumX", "momentumY",
            "energy", "vof", "solidEnergy"
        }

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
            rho = self.rho_fluid * torch.ones(coords.shape[0], 1, device=coords.device)
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

        # 流体区域PDE
        if "continuity" in self.enabledPdes:
            losses["continuity"] = self.continuity(rho, u, v, w, grads)
        if "momentumX" in self.enabledPdes:
            losses["momentumX"] = self.momX(rho, u, v, w, self.mu, grads, coords)
        if "momentumY" in self.enabledPdes:
            losses["momentumY"] = self.momY(rho, u, v, w, self.mu, grads, coords)
        if "energy" in self.enabledPdes and T is not None:
            losses["energy"] = self.energy(
                rho, self.cp_fluid, self.k_fluid, T, u, v, w, grads, coords
            )
        if "vof" in self.enabledPdes and alpha is not None:
            losses["vof"] = self.vof(alpha, u, v, w, grads)

        # 固体区域能量方程（纯导热）
        if "solidEnergy" in self.enabledPdes and T is not None:
            if self.solidMask is not None:
                solid_mask_t = torch.tensor(self.solidMask, dtype=torch.bool, device=coords.device)
                if solid_mask_t.any():
                    # 固体区域速度为0，只有扩散项
                    T_solid = T[solid_mask_t]
                    coords_solid = coords[solid_mask_t]
                    # 重新计算固体区域梯度
                    coords_solid = coords_solid.detach().requires_grad_(True)
                    fields_solid = self.model.predictFields(
                        coords_solid, t[:len(coords_solid)].detach().requires_grad_(True),
                        bcParams[:len(coords_solid)] if bcParams is not None else None
                    )
                    T_s = fields_solid.get("T") or fields_solid.get("field_4")
                    grads_s = fieldGradients({"T": T_s}, coords_solid, t[:len(coords_solid)])
                    losses["solidEnergy"] = self.solidEnergy(
                        self.rho_solid, self.cp_solid, self.k_solid,
                        T_s, grads_s, coords_solid
                    )
            # 如果没有提供mask，默认固体能量不激活（或者用VOF判断）
            elif alpha is not None:
                # 用VOF判断固体区域（alpha < 0 或者单独的场）
                pass

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
        # 1. 数据损失（匹配CFD结果）
        dLoss = self.dataLoss(predData, targetData)

        # 2. PDE物理损失
        pLosses = self.physicsLoss(coordsColloc, tColloc, bcParamsColloc)
        pTotal = sum(pLosses.values()) if pLosses else torch.tensor(0.0)

        # 3. 入口边界条件损失（强制UDF时序）
        bcTotal = torch.tensor(0.0, device=coordsData.device)
        bcBreakdown = {}
        if self.inletBC is not None:
            # 使用当前时间的代表值
            t_current = tColloc.mean().reshape(1, 1)
            if bcParamsColloc is not None:
                T_preheat = bcParamsColloc[:, 0:1].mean().reshape(1, 1)
                T_h2o2 = bcParamsColloc[:, 1:2].mean().reshape(1, 1)
                bcTotal, bcBreakdown = self.inletBC(self.model, t_current, T_preheat, T_h2o2)
            else:
                bcTotal, bcBreakdown = self.inletBC(self.model, t_current)

        # 总损失
        total = (
            self.lambdaData * dLoss
            + self.lambdaPhysics * pTotal
            + self.lambdaBc * bcTotal
        )

        breakdown = {
            "data": dLoss.item(),
            "physics": pTotal.item() if isinstance(pTotal, torch.Tensor) else 0.0,
            "bc": bcTotal.item() if isinstance(bcTotal, torch.Tensor) else 0.0,
        }
        for k, v in pLosses.items():
            breakdown[k] = v.item()
        breakdown.update(bcBreakdown)

        return total, breakdown
