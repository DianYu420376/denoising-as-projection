"""Reusable guidance helpers for diffusion sampling."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from cleandiffuser.utils import at_least_ndim

SUPPORTED_GUIDANCE_MODES = ("standard", "optimization", "hybrid")


def should_apply_optimization_guidance(
    guidance_mode: str,
    loop_i: int,
    last_steps: int,
    optimization_guidance_scale: float,
) -> bool:
    """True when optimization shift/backward is active on this reverse step."""
    return (
        guidance_mode in ("optimization", "hybrid")
        and optimization_guidance_scale != 0.0
        and apply_optimization_guidance_at_step(loop_i, last_steps)
    )


def should_apply_standard_classifier_guidance(
    guidance_mode: str,
    loop_i: int,
    last_steps: int,
    w_cg: float,
    use_optimization_guidance: bool,
) -> bool:
    """True when classifier guidance (w_cg) should modify the score on this step."""
    if w_cg == 0.0:
        return False
    if guidance_mode == "standard":
        return True
    if guidance_mode == "hybrid":
        return not use_optimization_guidance
    return False


def apply_optimization_guidance_at_step(
    loop_i: int,
    last_steps: int = 10,
) -> bool:
    """Whether to apply optimization guidance on reverse loop index ``loop_i``.

    ``loop_i`` runs from ``sample_steps`` (noisy) down to ``1`` (almost clean).
    With 20 steps and ``last_steps=10``: optimization on i in {1..10},
    standard unguided on i in {11..20}.
    """
    return loop_i <= last_steps


def validate_guidance_config(
    guidance_mode: str,
    w_cg: float,
    optimization_guidance_scale: float = 0.0,
) -> None:
    if guidance_mode not in SUPPORTED_GUIDANCE_MODES:
        raise ValueError(
            f"Unknown guidance_mode={guidance_mode!r}. "
            f"Supported modes: {SUPPORTED_GUIDANCE_MODES}."
        )
    if guidance_mode == "optimization" and w_cg != 0.0:
        raise ValueError(
            "guidance_mode='optimization' is incompatible with w_cg != 0. "
            "Optimization guidance evaluates E[x_0|x_t] at a shifted point and "
            "uses the native VP reverse step; set w_cg=0.0 or use guidance_mode='hybrid'."
        )
    if guidance_mode == "hybrid" and w_cg == 0.0 and optimization_guidance_scale == 0.0:
        raise ValueError(
            "guidance_mode='hybrid' requires at least one of w_cg or "
            "optimization_guidance_scale to be non-zero."
        )


def compute_reward_gradient(
    xt: torch.Tensor,
    t: torch.Tensor,
    classifier,
    condition_cg=None,
    fix_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Return detached reward gradient ∇_x log R(x_t) used for input shifting."""
    log_p, grad = classifier.gradients(xt.clone(), t, condition_cg)
    grad = grad.detach()
    if fix_mask is not None:
        grad = grad * (1.0 - fix_mask)
    return log_p, grad


def compute_optimization_shift(
    xt: torch.Tensor,
    grad: torch.Tensor,
    optimization_guidance_scale: float,
    fix_mask: Optional[torch.Tensor] = None,
    prior: Optional[torch.Tensor] = None,
    alpha: Optional[Union[torch.Tensor, float]] = None,
    sigma: Optional[Union[torch.Tensor, float]] = None,
) -> torch.Tensor:
    """Build x_t + η ∇R(x_t) for evaluating π_t at a reward-ascending point.

    When ``alpha`` and ``sigma`` are provided, the effective step size is
    ``optimization_guidance_scale * sigma**2 / alpha``.
    """
    scale = optimization_guidance_scale
    if alpha is not None and sigma is not None:
        alpha_b = at_least_ndim(alpha, xt.dim())
        sigma_b = at_least_ndim(sigma, xt.dim())
        scale = optimization_guidance_scale * (sigma_b ** 2) / alpha_b
    x_shift = xt + scale * grad
    if fix_mask is not None and prior is not None:
        x_shift = x_shift * (1.0 - fix_mask) + prior * fix_mask
    return x_shift


def compute_pi_t(
    x: torch.Tensor,
    pred: torch.Tensor,
    predict_noise: bool,
    alpha: Union[torch.Tensor, float],
    sigma: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Posterior mean π_t(x) = E[x_0 | x_t] under the VP forward process.

    CleanDiffuser uses x_t = α_t x_0 + σ_t ε, so
    E[x_0|x_t] = (x_t - σ_t ε_θ(x_t)) / α_t when the model predicts noise.
    """
    alpha_b = at_least_ndim(alpha, x.dim())
    sigma_b = at_least_ndim(sigma, x.dim())
    if predict_noise:
        return (x - sigma_b * pred) / alpha_b
    return pred


def vp_ddim_reverse_step(
    xt: torch.Tensor,
    eps_theta: torch.Tensor,
    alpha_curr: Union[torch.Tensor, float],
    sigma_curr: Union[torch.Tensor, float],
    alpha_prev: Union[torch.Tensor, float],
    sigma_prev: Union[torch.Tensor, float],
    std: Optional[Union[torch.Tensor, float]] = None,
    eta: float = 0.0,
    add_noise: bool = True,
) -> torch.Tensor:
    """Standard VP-DDIM reverse step with optional DDPM-style noise.

    Deterministic core (``eta=0``):
        x_{t-1} = (α_{t-1}/α_t) x_t + (σ_{t-1} - α_{t-1}σ_t/α_t) ε_θ(x_t)

    When ``eta > 0`` and ``std`` is provided, adds ``eta * std * z`` with ``z ~ N(0, I)``.
    ``std`` is the DDPM posterior std from ``DiscreteDiffusionSDE`` (``stds[i]``).
    """
    alpha_ratio = at_least_ndim(alpha_prev / alpha_curr, xt.dim())
    std_term = 0.0
    if std is not None and eta != 0.0:
        std_term = eta ** 2 * at_least_ndim(std, xt.dim()) ** 2
    mean = alpha_ratio * (xt - at_least_ndim(sigma_curr, xt.dim()) * eps_theta) + (
        at_least_ndim(sigma_prev, xt.dim()) ** 2 - std_term + 1e-8
    ).sqrt() * eps_theta
    if std is not None and eta != 0.0 and add_noise:
        mean = mean + eta * at_least_ndim(std, xt.dim()) * torch.randn_like(xt)
    return mean
    # xt = (
    #                     (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
    #                     (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta)
    #             if i > 1:
                    # xt += (stds[i] * torch.randn_like(xt))


def optimization_backward_step(
    xt: torch.Tensor,
    pi_t: torch.Tensor,
    alpha_curr: Union[torch.Tensor, float],
    sigma_curr: Union[torch.Tensor, float],
    alpha_prev: Union[torch.Tensor, float],
    sigma_prev: Union[torch.Tensor, float],
    std: Optional[Union[torch.Tensor, float]] = None,
    eta: float = 0.0,
    add_noise: bool = True,
) -> torch.Tensor:
    """Optimization backward step mixing chain ``x_t`` with ``π_t(x_eval)``.

    Deterministic core:
        x_{t-1} = (σ_{t-1}/σ_t) x_t + (α_{t-1} - σ_{t-1}α_t/σ_t) π_t(x_eval)

    ``std`` and ``eta`` are pass-through hooks for optional stochastic extensions
    (same convention as ``vp_ddim_reverse_step``).
    """
    std_term = 0.0
    if std is not None and eta != 0.0:
        std_term = eta ** 2 * at_least_ndim(std, xt.dim()) ** 2
    sigma_prev_prime = (at_least_ndim(sigma_prev, xt.dim()) ** 2 - std_term + 1e-8).sqrt()
    sigma_ratio = at_least_ndim(sigma_prev_prime / sigma_curr, xt.dim())
    coef_pi = at_least_ndim(alpha_prev, xt.dim()) - at_least_ndim(
        alpha_curr * sigma_ratio, xt.dim()
    )
    mean = sigma_ratio * xt + coef_pi * pi_t
    if std is not None and eta != 0.0 and add_noise:
        mean = mean + eta * at_least_ndim(std, xt.dim()) * torch.randn_like(xt)
    return mean


# Backward-compatible alias used by older call sites during migration.
def compute_guided_model_input(
    xt: torch.Tensor,
    t: torch.Tensor,
    classifier=None,
    condition_cg=None,
    fix_mask: Optional[torch.Tensor] = None,
    prior: Optional[torch.Tensor] = None,
    guidance_mode: str = "standard",
    optimization_guidance_scale: float = 0.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if (
        guidance_mode == "standard"
        or optimization_guidance_scale == 0.0
        or classifier is None
    ):
        return xt, None

    log_p, grad = compute_reward_gradient(xt, t, classifier, condition_cg, fix_mask)
    x_shift = compute_optimization_shift(
        xt, grad, optimization_guidance_scale, fix_mask, prior
    )
    return x_shift, log_p
