import torch


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
