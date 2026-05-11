"""
Multi-fluid PINN model — 3-phase Eulerian-Eulerian + species transport + solid heat.

Field layout (20 fields):  see config.py
Physics:  per-phase momentum, shared continuity, per-phase energy,
          species transport (phase 2, 3), VOF advection, k-ω turbulence,
          solid conduction.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from typing import Optional
from config import (
    F_U1, F_V1, F_W1, F_U2, F_V2, F_W2, F_U3, F_V3, F_W3,
    F_P, F_T, F_K, F_OMEGA, F_VOF2, F_VOF3,
    F_Y_H2O2_V, F_Y_H2O_V, F_Y_AIR_V,
    F_Y_H2O2_L, F_Y_H2O_L,
    SPECIES_SLICES, BULK_SPECIES,
)


# ── UDF inlet profiles ────────────────────────────────────────

def udf_inlet_temperature(
    t: torch.Tensor,
    inlet_name: str,
    T_preheat: float = 433.15,
    T_h2o2: float = 473.15,
    T_drying: float = 413.15,
    T_default: float = 300.0,
) -> torch.Tensor:
    T_target = torch.full_like(t, T_default)
    T_preheat_t = torch.as_tensor(T_preheat, dtype=t.dtype, device=t.device)
    T_h2o2_t = torch.as_tensor(T_h2o2, dtype=t.dtype, device=t.device)
    T_drying_t = torch.as_tensor(T_drying, dtype=t.dtype, device=t.device)

    if inlet_name == 'inlet_a':
        T_target = torch.where((t >= 0.32) & (t < 0.9), T_preheat_t, T_target)
        T_target = torch.where((t >= 2.12) & (t < 2.7), T_h2o2_t, T_target)
        T_target = torch.where(((t >= 4.82) & (t < 5.4)) | ((t >= 6.62) & (t < 7.4)), T_drying_t, T_target)
    elif inlet_name == 'inlet_b':
        T_target = torch.where((t >= 1.22) & (t < 1.8), T_preheat_t, T_target)
        T_target = torch.where((t >= 3.02) & (t < 3.6), T_h2o2_t, T_target)
        T_target = torch.where((t >= 5.72) & (t < 6.3), T_drying_t, T_target)
    return T_target.to(t.device)


def udf_inlet_species(
    t: torch.Tensor,
    inlet_name: str,
    Y_h2o2: float = 0.024989,
    Y_h2o: float = 0.152936,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (Y_h2o2, Y_h2o) mass fractions at inlet based on UDF timing."""
    Y_h2o2_val = torch.zeros_like(t)
    Y_h2o_val = torch.zeros_like(t)
    Y_h2o2_t = torch.as_tensor(Y_h2o2, dtype=t.dtype, device=t.device)
    Y_h2o_t = torch.as_tensor(Y_h2o, dtype=t.dtype, device=t.device)

    if inlet_name == 'inlet_a':
        mask = (t >= 2.12) & (t < 2.7)
    else:
        mask = (t >= 3.02) & (t < 3.6)

    Y_h2o2_val = torch.where(mask, Y_h2o2_t, Y_h2o2_val)
    Y_h2o_val = torch.where(mask, Y_h2o_t, Y_h2o_val)
    return Y_h2o2_val, Y_h2o_val


# ── Network ───────────────────────────────────────────────────

class PINNModel(nn.Module):
    def __init__(
        self,
        n_spatial: int = 3,
        hidden: int = 256,
        n_layers: int = 6,
        activation: str = "tanh",
        n_output: int = 20,
        parametric_bc: bool = True,
    ):
        super().__init__()
        self.parametric_bc = parametric_bc
        input_dim = n_spatial + 1 + (2 if parametric_bc else 0)

        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}.get(activation, nn.Tanh)

        layers = [nn.Linear(input_dim, hidden), act_cls()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), act_cls()]
        layers.append(nn.Linear(hidden, n_output))

        self.net = nn.Sequential(*layers)
        # Hard-BC masks (softplus for k,ω; sigmoid for VOF,species; identity for others)
        self.register_buffer("_field_activation", torch.zeros(n_output, dtype=torch.int32))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        bc_params: Optional[torch.Tensor] = None,
        hardT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.parametric_bc and bc_params is not None:
            if bc_params.shape[0] == 1 and len(x) > 1:
                bc_params = bc_params.expand(len(x), 2)
            inputs = torch.cat([x, t, bc_params], dim=-1)
        else:
            inputs = torch.cat([x, t], dim=-1)
        out = self.net(inputs)

        # Apply physical constraints (via column replacement to avoid in-place):
        #   VOF: sigmoid → [0,1]
        #   Species: sigmoid → [0,1]
        #   k, ω: softplus → (0, ∞)
        cols = [out[:, i:i + 1] for i in range(out.shape[1])]
        cols[F_VOF2] = torch.sigmoid(cols[F_VOF2])
        cols[F_VOF3] = torch.sigmoid(cols[F_VOF3])
        cols[F_K] = torch.nn.functional.softplus(cols[F_K]) + 1e-10
        cols[F_OMEGA] = torch.nn.functional.softplus(cols[F_OMEGA]) + 1e-10
        for idx in SPECIES_SLICES[2] + SPECIES_SLICES[3]:
            cols[idx] = torch.sigmoid(cols[idx])
        out = torch.cat(cols, dim=-1)

        if hardT is not None:
            cols = [out[:, i:i + 1] for i in range(out.shape[1])]
            cols[F_T] = hardT
            out = torch.cat(cols, dim=-1)
        return out


# ── Physics Loss ──────────────────────────────────────────────

class PhysicsLoss:
    @staticmethod
    def _safeScale(value: float) -> float:
        return max(abs(value), 1e-12)

    def __init__(
        self,
        # Phase 1
        rho_p1: float = 1.1455, mu_p1: float = 1.879e-5,
        cp_p1: float = 1006.0, k_p1: float = 0.026,
        # Phase 2
        rho_p2: float = 1.1455, mu_p2: float = 1.879e-5,
        cp_p2: float = 1500.0, k_p2: float = 0.02,
        # Phase 3
        rho_p3: float = 998.2, mu_p3: float = 1.003e-3,
        cp_p3: float = 4180.0, k_p3: float = 0.6,
        # Solid
        rho_solid: float = 2719.0, cp_solid: float = 871.0, k_solid: float = 202.4,
        # Species diffusivities
        D_h2o2_v: float = 2.5e-5, D_h2o_v: float = 2.5e-5,
        D_h2o2_l: float = 1.0e-9,
        # Drag
        K_drag_12: float = 1.0e4, K_drag_13: float = 5.0e4,
        # Control
        enabled: set | None = None,
        coord_scale: torch.Tensor | None = None,
        time_scale: float = 1.0,
        field_mean: torch.Tensor | None = None,
        field_std: torch.Tensor | None = None,
    ):
        # Store per-phase properties
        self.rho = {1: rho_p1, 2: rho_p2, 3: rho_p3}
        self.mu  = {1: mu_p1, 2: mu_p2, 3: mu_p3}
        self.cp  = {1: cp_p1, 2: cp_p2, 3: cp_p3}
        self.k   = {1: k_p1, 2: k_p2, 3: k_p3}
        self.rho_s, self.cp_s, self.k_s = rho_solid, cp_solid, k_solid
        self.D_v = {"h2o2": D_h2o2_v, "h2o": D_h2o_v}
        self.D_l = {"h2o2": D_h2o2_l}
        self.K12, self.K13 = K_drag_12, K_drag_13

        self.enabled = enabled or set()

        if coord_scale is not None:
            self.ls = coord_scale
        else:
            self.ls = torch.ones(3)
        self.dt = time_scale
        self.field_mean = field_mean
        self.field_std = field_std

        # Pre-computed scale factors
        self.hx = float(1.0 / self.ls[0]); self.hy = float(1.0 / self.ls[1]); self.hz = float(1.0 / self.ls[2])
        self.ht = float(1.0 / self.dt)
        self.hxx = float(1.0 / (self.ls[0] ** 2))
        self.hyy = float(1.0 / (self.ls[1] ** 2))
        self.hzz = float(1.0 / (self.ls[2] ** 2))

        # Field scale for PDE normalization (velocity, pressure, temperature)
        if field_std is not None:
            fS = field_std.detach().cpu().float()
            self.velScale = self._safeScale(float(fS[F_U1:F_U1 + 3].abs().max()))
            self.pScale = self._safeScale(float(fS[F_P].abs()))
            self.TScale = self._safeScale(float(fS[F_T].abs()))
        else:
            self.velScale = self.pScale = self.TScale = 1.0

        self._ls_abs_mean = self._safeScale(float(self.ls.detach().cpu().float().abs().mean()))
        self._ts_abs = self._safeScale(float(abs(self.dt)))

        # Pre-compute PDE scales
        self._compute_pde_scales()

    def _compute_pde_scales(self):
        L = self._ls_abs_mean
        V = self.velScale; P = self.pScale
        T_abs = self._ts_abs

        self.contScale  = self._safeScale(V / L)
        self.momScale1  = self._safeScale(self.rho[1] * (V / T_abs + V**2 / L) + P / L + self.mu[1] * V / L**2)
        self.momScale2  = self._safeScale(self.rho[2] * (V / T_abs + V**2 / L) + P / L + self.mu[2] * V / L**2)
        self.momScale3  = self._safeScale(self.rho[3] * (V / T_abs + V**2 / L) + P / L + self.mu[3] * V / L**2)
        self.enScale2   = self._safeScale(self.rho[2] * self.cp[2] * (self.TScale / T_abs + V * self.TScale / L) + self.k[2] * self.TScale / L**2)
        self.enScale3   = self._safeScale(self.rho[3] * self.cp[3] * (self.TScale / T_abs + V * self.TScale / L) + self.k[3] * self.TScale / L**2)
        self.enSolidS   = self._safeScale(self.rho_s * self.cp_s * self.TScale / T_abs + self.k_s * self.TScale / L**2)
        self.spScale2   = self._safeScale(self.rho[2] / T_abs + self.rho[2] * V / L + self.rho[2] * 2.5e-5 / L**2)
        self.spScale3   = self._safeScale(self.rho[3] / T_abs + self.rho[3] * V / L + self.rho[3] * 1.0e-9 / L**2)
        # Turbulence scales (simplified)
        self.kScale  = self._safeScale(self.rho[1] / T_abs)
        self.wScale  = self._safeScale(self.rho[1] / T_abs)

    # ── derivative helpers ────────────────────────────────

    def _d1(self, u: torch.Tensor, x: torch.Tensor, idx: int) -> torch.Tensor:
        du = grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        scale = [self.hx, self.hy, self.hz][idx]
        return du[:, idx:idx + 1] * scale

    def _d1t(self, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0] * self.ht

    def _d2(self, u: torch.Tensor, x: torch.Tensor, idx: int) -> torch.Tensor:
        du = grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        du_i = du[:, idx:idx + 1]
        d2u = grad(du_i, x, torch.ones_like(du_i), create_graph=True, retain_graph=True)[0]
        scale = [self.hxx, self.hyy, self.hzz][idx]
        return d2u[:, idx:idx + 1] * scale

    def _laplacian(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._d2(u, x, 0) + self._d2(u, x, 1) + self._d2(u, x, 2)

    def _advect(self, u: torch.Tensor, phi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return u * self._d1(phi, x, 0) + self._d1(phi, x, 1) + self._d1(phi, x, 2)

    # ── per-phase velocity extraction ─────────────────────

    def _vel(self, out: torch.Tensor, phase: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if phase == 1: u, v, w = F_U1, F_V1, F_W1
        elif phase == 2: u, v, w = F_U2, F_V2, F_W2
        else: u, v, w = F_U3, F_V3, F_W3
        return out[:, u:u + 1], out[:, v:v + 1], out[:, w:w + 1]

    # ── main compute ──────────────────────────────────────

    def compute(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        out: torch.Tensor,
        solid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all PDE residuals. Returns dict of name → residual."""
        # Denormalize output for physical computation
        if self.field_mean is not None and self.field_std is not None:
            fm = self.field_mean.to(out.device)
            fs = self.field_std.to(out.device)
            out_phys = out * fs + fm
        else:
            out_phys = out

        res = {}

        # ── Phase 1 velocity ──
        u1, v1, w1 = self._vel(out_phys, 1)
        # ── Phase 2 velocity ──
        u2, v2, w2 = self._vel(out_phys, 2)
        # ── Phase 3 velocity ──
        u3, v3, w3 = self._vel(out_phys, 3)

        p  = out_phys[:, F_P:F_P + 1]
        T  = out_phys[:, F_T:F_T + 1]
        k_field  = out_phys[:, F_K:F_K + 1]
        w_field  = out_phys[:, F_OMEGA:F_OMEGA + 1]
        vof2 = out_phys[:, F_VOF2:F_VOF2 + 1]
        vof3 = out_phys[:, F_VOF3:F_VOF3 + 1]

        # ── Phase 1 momentum + continuity ──
        if "continuity" in self.enabled:
            lap_u1 = self._laplacian(u1, x)
            lap_v1 = self._laplacian(v1, x)
            lap_w1 = self._laplacian(w1, x)
            dudx = self._d1(u1, x, 0); dvdy = self._d1(v1, x, 1); dwdz = self._d1(w1, x, 2)
            res["continuity"] = (dudx + dvdy + dwdz) / self.contScale

        for name, u, v, w, rho_p, mu_p, en in [
            ("momentum_p1", u1, v1, w1, self.rho[1], self.mu[1], "momentum_p1"),
            ("momentum_p2", u2, v2, w2, self.rho[2], self.mu[2], "momentum_p2"),
            ("momentum_p3", u3, v3, w3, self.rho[3], self.mu[3], "momentum_p3"),
        ]:
            ms = {"momentum_p1": self.momScale1, "momentum_p2": self.momScale2, "momentum_p3": self.momScale3}[en]
            for ax, vel, label in [
                ("x", u, f"{name}_x"), ("y", v, f"{name}_y"), ("z", w, f"{name}_z")
            ]:
                if label not in self.enabled:
                    continue
                lap = self._laplacian(vel, x)
                dveldt = self._d1t(vel, t)
                adv = self._advect(u, vel, x) + self._advect(v, vel, x) + self._advect(w, vel, x)
                dp = self._d1(p, x, 0) if ax == "x" else self._d1(p, x, 1) if ax == "y" else self._d1(p, x, 2)
                res[label] = (rho_p * (dveldt + adv) + dp - mu_p * lap) / ms

        # ── Phase energy equations ──
        for label_name, vel_u, vel_v, vel_w, rho_p, cp_p, k_p, en_s in [
            ("energy_p2", u2, v2, w2, self.rho[2], self.cp[2], self.k[2], self.enScale2),
            ("energy_p3", u3, v3, w3, self.rho[3], self.cp[3], self.k[3], self.enScale3),
        ]:
            if label_name not in self.enabled:
                continue
            dTdt = self._d1t(T, t)
            adv_T = self._advect(vel_u, T, x) + self._advect(vel_v, T, x) + self._advect(vel_w, T, x)
            lap_T = self._laplacian(T, x)
            res[label_name] = (rho_p * cp_p * (dTdt + adv_T) - k_p * lap_T) / en_s

        # ── Solid energy ──
        if "energy_solid" in self.enabled:
            dTdt = self._d1t(T, t)
            lap_T = self._laplacian(T, x)
            res["energy_solid"] = (self.rho_s * self.cp_s * dTdt - self.k_s * lap_T) / self.enSolidS

        # ── Species transport (Phase 2 — vapor) ──
        Y_h2o2_v = out_phys[:, F_Y_H2O2_V:F_Y_H2O2_V + 1]
        Y_h2o_v  = out_phys[:, F_Y_H2O_V:F_Y_H2O_V + 1]
        for label_name, Y, D in [
            ("species_h2o2_v", Y_h2o2_v, self.D_v["h2o2"]),
            ("species_h2o_v",  Y_h2o_v,  self.D_v["h2o"]),
        ]:
            if label_name not in self.enabled:
                continue
            dYdt = self._d1t(Y, t)
            adv_Y = self._advect(u2, Y, x) + self._advect(v2, Y, x) + self._advect(w2, Y, x)
            lap_Y = self._laplacian(Y, x)
            res[label_name] = (self.rho[2] * (dYdt + adv_Y) - self.rho[2] * D * lap_Y) / self.spScale2

        # ── Species transport (Phase 3 — liquid) ──
        Y_h2o2_l = out_phys[:, F_Y_H2O2_L:F_Y_H2O2_L + 1]
        if "species_h2o2_l" in self.enabled:
            dYdt = self._d1t(Y_h2o2_l, t)
            adv_Y = self._advect(u3, Y_h2o2_l, x) + self._advect(v3, Y_h2o2_l, x) + self._advect(w3, Y_h2o2_l, x)
            lap_Y = self._laplacian(Y_h2o2_l, x)
            res["species_h2o2_l"] = (self.rho[3] * (dYdt + adv_Y) - self.rho[3] * self.D_l["h2o2"] * lap_Y) / self.spScale3

        # ── VOF advection ──
        if "vof_advection" in self.enabled:
            dvof2dt = self._d1t(vof2, t)
            adv_vof2 = self._advect(u2, vof2, x) + self._advect(v2, vof2, x) + self._advect(w2, vof2, x)
            res["vof_advection"] = dvof2dt + adv_vof2  # dimensionless, no scaling needed

        # ── k-ω turbulence (simplified — modelled as transport) ──
        if "k_transport" in self.enabled:
            dkdt = self._d1t(k_field, t)
            adv_k = self._advect(u1, k_field, x) + self._advect(v1, k_field, x) + self._advect(w1, k_field, x)
            lap_k = self._laplacian(k_field, x)
            res["k_transport"] = (self.rho[1] * (dkdt + adv_k) - (self.mu[1] + self.mu[1] * 10) * lap_k) / self.kScale

        if "omega_transport" in self.enabled:
            dwdt = self._d1t(w_field, t)
            adv_w = self._advect(u1, w_field, x) + self._advect(v1, w_field, x) + self._advect(w1, w_field, x)
            lap_w = self._laplacian(w_field, x)
            res["omega_transport"] = (self.rho[1] * (dwdt + adv_w) - (self.mu[1] + self.mu[1] * 10) * lap_w) / self.wScale

        return res


# ── Inlet BC Loss (multi-fluid) ──────────────────────────────

class InletBCLoss:
    def __init__(
        self,
        inlet_a_coords: np.ndarray,
        inlet_b_coords: np.ndarray,
        coord_min: np.ndarray,
        coord_max: np.ndarray,
        time_min: float = 0.0,
        time_max: float = 1.0,
        T_min: float = 273.15,
        T_max: float = 573.15,
        device: str = "cpu",
        hard_inlet_bc: bool = False,
    ):
        self.device = device
        self.hard_inlet_bc = hard_inlet_bc

        coord_range = coord_max - coord_min + 1e-12
        inlet_a_norm = (inlet_a_coords - coord_min) / coord_range
        inlet_b_norm = (inlet_b_coords - coord_min) / coord_range

        self.inlet_a = torch.tensor(inlet_a_norm, dtype=torch.float32, device=device)
        self.inlet_b = torch.tensor(inlet_b_norm, dtype=torch.float32, device=device)

        self.time_min = time_min
        self.time_range = time_max - time_min + 1e-12
        self.T_min = T_min
        self.T_range = T_max - T_min + 1e-12

    def __call__(
        self,
        model: PINNModel,
        t_norm: torch.Tensor,
        bc_params: torch.Tensor,
        T_target_a_norm: torch.Tensor,
        T_target_b_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        n_samples = t_norm.shape[0]
        t_phys = t_norm * self.time_range + self.time_min

        inlet_a = self.inlet_a.repeat(n_samples, 1)
        inlet_b = self.inlet_b.repeat(n_samples, 1)
        t_a = t_norm.repeat_interleave(len(self.inlet_a), dim=0)
        t_b = t_norm.repeat_interleave(len(self.inlet_b), dim=0)
        bc_params_a = bc_params.repeat_interleave(len(self.inlet_a), dim=0)
        bc_params_b = bc_params.repeat_interleave(len(self.inlet_b), dim=0)

        T_target_a_expand = T_target_a_norm.repeat_interleave(len(self.inlet_a), dim=0)
        T_target_b_expand = T_target_b_norm.repeat_interleave(len(self.inlet_b), dim=0)

        if self.hard_inlet_bc:
            pred_a = model(inlet_a, t_a, bc_params_a, hardT=T_target_a_expand)
            pred_b = model(inlet_b, t_b, bc_params_b, hardT=T_target_b_expand)
        else:
            pred_a = model(inlet_a, t_a, bc_params_a)
            pred_b = model(inlet_b, t_b, bc_params_b)

        T_pred_a = pred_a[:, F_T:F_T + 1]
        T_pred_b = pred_b[:, F_T:F_T + 1]

        loss = nn.functional.mse_loss(T_pred_a, T_target_a_expand) + \
               nn.functional.mse_loss(T_pred_b, T_target_b_expand)

        metrics = {
            'bc_inlet_a': nn.functional.mse_loss(T_pred_a, T_target_a_expand).item(),
            'bc_inlet_b': nn.functional.mse_loss(T_pred_b, T_target_b_expand).item(),
        }

        if not self.hard_inlet_bc:
            T_err_a = (T_pred_a - T_target_a_expand) * self.T_range
            T_err_b = (T_pred_b - T_target_b_expand) * self.T_range
            metrics.update({
                'T_inlet_a_rmse_K': torch.sqrt((T_err_a ** 2).mean()).item(),
                'T_inlet_b_rmse_K': torch.sqrt((T_err_b ** 2).mean()).item(),
            })

        return loss, metrics
