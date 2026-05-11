import torch
import torch.nn as nn


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def fieldGradients(
    fields: dict[str, torch.Tensor],
    coords: torch.Tensor,
    t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, field in fields.items():
        spatialGrad = gradient(field, coords)
        grads[f"d{name}_dx"] = spatialGrad[:, 0:1]
        grads[f"d{name}_dy"] = spatialGrad[:, 1:2]
        if coords.shape[1] > 2:
            grads[f"d{name}_dz"] = spatialGrad[:, 2:3]
        grads[f"d{name}_dt"] = gradient(field, t)
    return grads


def secondDerivatives(
    firstDeriv: torch.Tensor,
    coords: torch.Tensor,
    component: int,
) -> torch.Tensor:
    return gradient(firstDeriv, coords)[:, component : component + 1]


class continuityLoss:
    def __call__(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor | None,
        grads: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        res = grads["drho_dt"] + rho * (grads["du_dx"] + grads["dv_dy"])
        res = res + u * grads["drho_dx"] + v * grads["drho_dy"]
        if w is not None and "dw_dz" in grads:
            res = res + rho * grads["dw_dz"] + w * grads["drho_dz"]
        return (res ** 2).mean()


class momentumXLoss:
    def __call__(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor | None,
        mu: float,
        grads: dict[str, torch.Tensor],
        coords: torch.Tensor,
    ) -> torch.Tensor:
        convection = u * grads["du_dx"] + v * grads["du_dy"]
        if w is not None and "du_dz" in grads:
            convection = convection + w * grads["du_dz"]
        d2u_dx2 = secondDerivatives(grads["du_dx"], coords, 0)
        d2u_dy2 = secondDerivatives(grads["du_dy"], coords, 1)
        diffusion = mu * (d2u_dx2 + d2u_dy2)
        if coords.shape[1] > 2 and "du_dz" in grads:
            d2u_dz2 = secondDerivatives(grads["du_dz"], coords, 2)
            diffusion = diffusion + mu * d2u_dz2
        dp_dx = grads.get("dp_dx", torch.zeros_like(u))
        res = rho * (grads["du_dt"] + convection) + dp_dx - diffusion
        return (res ** 2).mean()


class momentumYLoss:
    def __call__(
        self,
        rho: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor | None,
        mu: float,
        grads: dict[str, torch.Tensor],
        coords: torch.Tensor,
    ) -> torch.Tensor:
        convection = u * grads["dv_dx"] + v * grads["dv_dy"]
        if w is not None and "dv_dz" in grads:
            convection = convection + w * grads["dv_dz"]
        d2v_dx2 = secondDerivatives(grads["dv_dx"], coords, 0)
        d2v_dy2 = secondDerivatives(grads["dv_dy"], coords, 1)
        diffusion = mu * (d2v_dx2 + d2v_dy2)
        if coords.shape[1] > 2 and "dv_dz" in grads:
            d2v_dz2 = secondDerivatives(grads["dv_dz"], coords, 2)
            diffusion = diffusion + mu * d2v_dz2
        dp_dy = grads.get("dp_dy", torch.zeros_like(v))
        res = rho * (grads["dv_dt"] + convection) + dp_dy - diffusion
        return (res ** 2).mean()


class energyLoss:
    def __call__(
        self,
        rho: torch.Tensor,
        cp: float,
        k: float,
        T: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor | None,
        grads: dict[str, torch.Tensor],
        coords: torch.Tensor,
    ) -> torch.Tensor:
        convection = u * grads["dT_dx"] + v * grads["dT_dy"]
        if w is not None and "dT_dz" in grads:
            convection = convection + w * grads["dT_dz"]
        d2T_dx2 = secondDerivatives(grads["dT_dx"], coords, 0)
        d2T_dy2 = secondDerivatives(grads["dT_dy"], coords, 1)
        diffusion = k * (d2T_dx2 + d2T_dy2)
        if coords.shape[1] > 2 and "dT_dz" in grads:
            d2T_dz2 = secondDerivatives(grads["dT_dz"], coords, 2)
            diffusion = diffusion + k * d2T_dz2
        res = rho * cp * (grads["dT_dt"] + convection) - diffusion
        return (res ** 2).mean()


class vofLoss:
    def __call__(
        self,
        alpha: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor | None,
        grads: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        res = grads["dalpha_dt"] + u * grads["dalpha_dx"] + v * grads["dalpha_dy"]
        res = res + alpha * (grads["du_dx"] + grads["dv_dy"])
        if w is not None and "dalpha_dz" in grads and "dw_dz" in grads:
            res = res + w * grads["dalpha_dz"] + alpha * grads["dw_dz"]
        return (res ** 2).mean()


# ============================================================
# 固体区域能量方程（只有导热，无对流）
# ============================================================
class solidEnergyLoss:
    """固体导热方程：rho_s * cp_s * dT/dt = k_s * ∇²T"""
    def __call__(
        self,
        rho_s: float,
        cp_s: float,
        k_s: float,
        T: torch.Tensor,
        grads: dict[str, torch.Tensor],
        coords: torch.Tensor,
    ) -> torch.Tensor:
        d2T_dx2 = secondDerivatives(grads["dT_dx"], coords, 0)
        d2T_dy2 = secondDerivatives(grads["dT_dy"], coords, 1)
        diffusion = k_s * (d2T_dx2 + d2T_dy2)
        if coords.shape[1] > 2 and "dT_dz" in grads:
            d2T_dz2 = secondDerivatives(grads["dT_dz"], coords, 2)
            diffusion = diffusion + k_s * d2T_dz2
        res = rho_s * cp_s * grads["dT_dt"] - diffusion
        return (res ** 2).mean()


# ============================================================
# UDF 时序边界条件函数
# ============================================================
def udf_inlet_temperature(t: torch.Tensor, inlet_name: str) -> torch.Tensor:
    """
    根据UDF返回对应时间的入口温度
    t: 时间 [s], shape: [N, 1]
    inlet_name: 'inlet_a' 或 'inlet_b'
    返回: 温度 [K], shape: [N, 1]
    """
    T_DEFAULT = 300.0      # 常温
    T_PREHEAT = 433.15     # 160°C - 预热气体温度
    T_H2O2 = 473.15        # 200°C - 双氧水温度
    T_DRYING = 413.15      # 140°C

    t_np = t.detach().cpu().numpy().flatten()
    T_target = torch.ones_like(t) * T_DEFAULT

    if inlet_name == 'inlet_a':
        # inlet_a UDF时序:
        # t < 0.32s: 300K
        # 0.32-0.9s: T_PREHEAT 160°C
        # 0.9-2.12s: 300K
        # 2.12-2.7s: T_H2O2 200°C
        # 2.7-4.82s: 300K
        # 4.82-5.4s: T_DRYING 140°C
        for i, ti in enumerate(t_np):
            if 0.32 <= ti < 0.9:
                T_target[i] = T_PREHEAT
            elif 2.12 <= ti < 2.7:
                T_target[i] = T_H2O2
            elif 4.82 <= ti < 5.4:
                T_target[i] = T_DRYING
            elif 6.62 <= ti < 7.4:
                T_target[i] = T_DRYING

    elif inlet_name == 'inlet_b':
        # inlet_b UDF时序:
        # t < 1.22s: 300K
        # 1.22-1.8s: T_PREHEAT 160°C
        # 1.8-3.02s: 300K
        # 3.02-3.6s: T_H2O2 200°C
        # 3.6-5.72s: 300K
        # 5.72-6.3s: T_DRYING 140°C
        for i, ti in enumerate(t_np):
            if 1.22 <= ti < 1.8:
                T_target[i] = T_PREHEAT
            elif 3.02 <= ti < 3.6:
                T_target[i] = T_H2O2
            elif 5.72 <= ti < 6.3:
                T_target[i] = T_DRYING

    return T_target.to(t.device)


# ============================================================
# 入口边界条件损失
# ============================================================
class inletBCLoss:
    """
    在入口面坐标上强制约束温度边界条件
    支持参数化：传入 T_preheat 和 T_h2o2 作为可学习/可变参数
    """
    def __init__(
        self,
        inlet_a_coords: torch.Tensor,  # [N_a, 3]
        inlet_b_coords: torch.Tensor,  # [N_b, 3]
        use_parametric: bool = True,   # 是否使用参数化入口温度
    ):
        self.inlet_a_coords = inlet_a_coords
        self.inlet_b_coords = inlet_b_coords
        self.use_parametric = use_parametric

    def __call__(
        self,
        model: nn.Module,
        t: torch.Tensor,           # 当前时间 [1, 1] 或 [N, 1]
        T_preheat: torch.Tensor | None = None,  # 参数化预热温度 [1, 1]
        T_h2o2: torch.Tensor | None = None,     # 参数化双氧水温度 [1, 1]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        N_a = len(self.inlet_a_coords)
        N_b = len(self.inlet_b_coords)

        # 扩展 t 到所有入口面
        t_a = t.expand(N_a, 1).contiguous().requires_grad_(True)
        t_b = t.expand(N_b, 1).contiguous().requires_grad_(True)

        coords_a = self.inlet_a_coords.contiguous().requires_grad_(True)
        coords_b = self.inlet_b_coords.contiguous().requires_grad_(True)

        # 预测入口温度
        if self.use_parametric and T_preheat is not None and T_h2o2 is not None:
            # 参数化模式：bcParams = [T_preheat, T_h2o2]
            bc_params_a = torch.cat([T_preheat, T_h2o2], dim=-1).expand(N_a, 2)
            bc_params_b = torch.cat([T_preheat, T_h2o2], dim=-1).expand(N_b, 2)

            fields_a = model.predictFields(coords_a, t_a, bc_params_a)
            fields_b = model.predictFields(coords_b, t_b, bc_params_b)
        else:
            # 直接使用UDF时序模式
            fields_a = model.predictFields(coords_a, t_a)
            fields_b = model.predictFields(coords_b, t_b)

        # 提取温度场
        T_a = fields_a.get("T") or fields_a.get("field_4")
        T_b = fields_b.get("T") or fields_b.get("field_4")

        # 目标温度（来自UDF时序）
        T_target_a = udf_inlet_temperature(t_a, 'inlet_a')
        T_target_b = udf_inlet_temperature(t_b, 'inlet_b')

        loss_a = nn.functional.mse_loss(T_a, T_target_a)
        loss_b = nn.functional.mse_loss(T_b, T_target_b)

        total_loss = loss_a + loss_b
        breakdown = {"bc_inlet_a": loss_a.item(), "bc_inlet_b": loss_b.item()}

        return total_loss, breakdown
