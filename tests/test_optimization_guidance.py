import pytest
import torch
import torch.nn as nn

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.diffusion.guidance import (
    apply_optimization_guidance_at_step,
    compute_optimization_shift,
    compute_pi_t,
    compute_reward_gradient,
    optimization_backward_step,
    should_apply_optimization_guidance,
    should_apply_standard_classifier_guidance,
    validate_guidance_config,
    vp_ddim_reverse_step,
)
from cleandiffuser.nn_diffusion import DiT1d


DEVICE = "cpu"
HORIZON = 5
OBS_DIM = 4
ACT_DIM = 2
DIM = OBS_DIM + ACT_DIM
TOL = 1e-5


class _MockReturnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, t, c=None):
        return self.scale * x.reshape(x.shape[0], -1).sum(dim=1, keepdim=True)


class MockReturnClassifier(BaseClassifier):
    def __init__(self, device=DEVICE):
        super().__init__(_MockReturnNet(), device=device)

    def logp(self, x, noise, c=None):
        return self.model_ema(x, noise, c)

    def loss(self, x, noise, y):
        pred = self.model(x, noise, c=None)
        return ((pred - y) ** 2).mean()


@pytest.fixture
def planning_setup():
    nn_diffusion = DiT1d(
        DIM,
        emb_dim=32,
        d_model=64,
        n_heads=4,
        depth=2,
        timestep_emb_type="fourier",
    )
    fix_mask = torch.zeros((HORIZON, DIM))
    fix_mask[0, :OBS_DIM] = 1.0
    loss_weight = torch.ones((HORIZON, DIM))

    agent = DiscreteDiffusionSDE(
        nn_diffusion=nn_diffusion,
        nn_condition=None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=MockReturnClassifier(device=DEVICE),
        diffusion_steps=5,
        device=DEVICE,
    )
    agent.eval()
    return agent, fix_mask


def _make_prior(batch_size=2):
    prior = torch.randn(batch_size, HORIZON, DIM)
    prior[:, 0, :OBS_DIM] = torch.linspace(1.0, 4.0, OBS_DIM)
    return prior


def test_apply_optimization_guidance_at_step_last_half():
    assert not apply_optimization_guidance_at_step(20, last_steps=10)
    assert not apply_optimization_guidance_at_step(11, last_steps=10)
    assert apply_optimization_guidance_at_step(10, last_steps=10)
    assert apply_optimization_guidance_at_step(1, last_steps=10)


def test_optimization_backward_step_matches_pdf_sigma_ratio():
    xt = torch.tensor([[[1.0, 0.0]]])
    pi_t = torch.tensor([[[2.0, 1.0]]])
    alpha_curr = torch.tensor(0.8)
    sigma_curr = torch.tensor(0.6)
    alpha_prev = torch.tensor(0.9)
    sigma_prev = torch.tensor(0.435889894)
    step = optimization_backward_step(
        xt, pi_t, alpha_curr, sigma_curr, alpha_prev, sigma_prev
    )
    sigma_ratio = sigma_prev / sigma_curr
    coef_pi = alpha_prev - sigma_prev * alpha_curr / sigma_curr
    expected = sigma_ratio * xt + coef_pi * pi_t
    assert torch.allclose(step, expected)


def test_optimization_backward_uses_xt_and_pi_not_collapsed():
    xt = torch.tensor([[[1.0, 0.0]]])
    x_shift = torch.tensor([[[3.0, 2.0]]])
    eps_shift = torch.tensor([[[0.5, -0.25]]])
    alpha_curr = torch.tensor(0.8)
    sigma_curr = torch.tensor(0.6)
    alpha_prev = torch.tensor(0.9)
    sigma_prev = torch.tensor(0.435889894)
    pi_shift = compute_pi_t(x_shift, eps_shift, True, alpha_curr, sigma_curr)
    step = optimization_backward_step(
        xt, pi_shift, alpha_curr, sigma_curr, alpha_prev, sigma_prev
    )
    sigma_ratio = sigma_prev / sigma_curr
    coef_pi = alpha_prev - sigma_prev * alpha_curr / sigma_curr
    collapsed = (sigma_prev / sigma_curr) * x_shift + coef_pi * pi_shift
    assert not torch.allclose(step, collapsed, atol=TOL)
    assert torch.allclose(step, sigma_ratio * xt + coef_pi * pi_shift, atol=TOL)


def test_vp_ddim_reverse_step_mixes_xt_with_eps():
    xt = torch.tensor([[[1.0, 0.0]]])
    eps = torch.tensor([[[0.5, -0.25]]])
    alpha_curr = torch.tensor(0.8)
    sigma_curr = torch.tensor(0.6)
    alpha_prev = torch.tensor(0.9)
    sigma_prev = torch.tensor(0.435889894)
    step = vp_ddim_reverse_step(xt, eps, alpha_curr, sigma_curr, alpha_prev, sigma_prev)
    alpha_ratio = alpha_prev / alpha_curr
    coef_eps = sigma_prev - alpha_prev * sigma_curr / alpha_curr
    expected = alpha_ratio * xt + coef_eps * eps
    assert torch.allclose(step, expected)


def test_vp_ddim_matches_collapsed_form_when_eval_at_xt():
    xt = torch.tensor([[[1.0, 0.0]]])
    eps = torch.tensor([[[0.5, -0.25]]])
    alpha_curr = torch.tensor(0.8)
    sigma_curr = torch.tensor(0.6)
    alpha_prev = torch.tensor(0.9)
    sigma_prev = torch.tensor(0.435889894)
    x_theta = compute_pi_t(xt, eps, True, alpha_curr, sigma_curr)
    step = vp_ddim_reverse_step(xt, eps, alpha_curr, sigma_curr, alpha_prev, sigma_prev)
    collapsed = alpha_prev * x_theta + sigma_prev * eps
    assert torch.allclose(step, collapsed, atol=TOL)


def test_vp_ddim_optimization_step_uses_chain_xt():
    xt = torch.tensor([[[1.0, 0.0]]])
    x_shift = torch.tensor([[[3.0, 2.0]]])
    eps_shift = torch.tensor([[[0.5, -0.25]]])
    alpha_curr = torch.tensor(0.8)
    sigma_curr = torch.tensor(0.6)
    alpha_prev = torch.tensor(0.9)
    sigma_prev = torch.tensor(0.435889894)
    pi_shift = compute_pi_t(x_shift, eps_shift, True, alpha_curr, sigma_curr)
    step = optimization_backward_step(
        xt, pi_shift, alpha_curr, sigma_curr, alpha_prev, sigma_prev
    )
    wrong = alpha_prev * pi_shift + sigma_prev * eps_shift
    assert not torch.allclose(step, wrong, atol=TOL)


def test_validate_guidance_config_conflict():
    with pytest.raises(ValueError, match="incompatible with w_cg"):
        validate_guidance_config("optimization", w_cg=0.3)


def test_validate_guidance_config_hybrid_allows_both():
    validate_guidance_config("hybrid", w_cg=0.3, optimization_guidance_scale=0.3)


def test_validate_guidance_config_hybrid_requires_one_guidance():
    with pytest.raises(ValueError, match="requires at least one"):
        validate_guidance_config("hybrid", w_cg=0.0, optimization_guidance_scale=0.0)


def test_compute_optimization_shift_alpha_sigma_scale():
    xt = torch.zeros((1, 2, 4))
    grad = torch.ones((1, 2, 4))
    alpha = torch.tensor(0.8)
    sigma = torch.tensor(0.6)
    out = compute_optimization_shift(xt, grad, 0.3, alpha=alpha, sigma=sigma)
    expected_scale = 0.3 * (0.6 ** 2) / 0.8
    assert torch.allclose(out, expected_scale * grad)


def test_compute_pi_t_noise_prediction():
    x = torch.tensor([[[2.0, 1.0]]])
    eps = torch.tensor([[[0.5, -0.25]]])
    alpha = torch.tensor(0.8)
    sigma = torch.tensor(0.6)
    pi = compute_pi_t(x, eps, True, alpha, sigma)
    expected = (x - sigma * eps) / alpha
    assert torch.allclose(pi, expected)


def test_compute_pi_t_x0_prediction():
    x = torch.randn(1, HORIZON, DIM)
    x0 = torch.randn(1, HORIZON, DIM)
    pi = compute_pi_t(x, x0, False, 0.8, 0.6)
    assert torch.allclose(pi, x0)


def test_compute_optimization_shift_chain_invariant():
    xt = torch.randn(1, HORIZON, DIM)
    xt_copy = xt.clone()
    t = torch.zeros(1, dtype=torch.long)
    prior = _make_prior(1)
    fix_mask = torch.zeros((HORIZON, DIM))
    fix_mask[0, :OBS_DIM] = 1.0
    classifier = MockReturnClassifier(device=DEVICE)

    _, grad = compute_reward_gradient(xt, t, classifier, fix_mask=fix_mask)
    compute_optimization_shift(xt, grad, 0.1, fix_mask, prior)

    assert torch.allclose(xt, xt_copy)


def test_standard_sampling_unchanged_with_defaults(planning_setup):
    agent, _ = planning_setup
    prior = _make_prior(2)
    torch.manual_seed(0)

    traj_a, _ = agent.sample(
        prior,
        solver="ddpm",
        n_samples=2,
        sample_steps=5,
        use_ema=True,
        w_cg=0.0,
        guidance_mode="standard",
        optimization_guidance_scale=0.0,
    )

    torch.manual_seed(0)
    traj_b, _ = agent.sample(
        prior,
        solver="ddpm",
        n_samples=2,
        sample_steps=5,
        use_ema=True,
        w_cg=0.0,
    )

    assert traj_a.shape == prior.shape
    assert torch.allclose(traj_a, traj_b, atol=TOL)


def test_standard_w_cg_changes_output(planning_setup):
    agent, _ = planning_setup
    prior = _make_prior(2)

    torch.manual_seed(1)
    traj_none, _ = agent.sample(
        prior,
        solver="ddpm",
        n_samples=2,
        sample_steps=5,
        use_ema=True,
        w_cg=0.0,
        guidance_mode="standard",
    )

    torch.manual_seed(1)
    traj_cg, _ = agent.sample(
        prior,
        solver="ddpm",
        n_samples=2,
        sample_steps=5,
        use_ema=True,
        w_cg=0.3,
        guidance_mode="standard",
    )

    assert not torch.allclose(traj_none, traj_cg, atol=TOL)


def test_optimization_sampling_runs(planning_setup):
    agent, _ = planning_setup
    prior = _make_prior(2)

    traj, log = agent.sample(
        prior,
        solver="ddpm",
        n_samples=2,
        sample_steps=5,
        use_ema=True,
        w_cg=0.0,
        guidance_mode="optimization",
        optimization_guidance_scale=0.05,
    )

    assert traj.shape == prior.shape
    assert log["log_p"] is not None


def test_optimization_pi_evaluated_at_shift(planning_setup):
    agent, fix_mask = planning_setup
    prior = _make_prior(1)
    xt = torch.randn_like(prior)
    t = torch.zeros(1, dtype=torch.long)
    model = agent.model_ema
    alpha = agent.alpha[1]
    sigma = agent.sigma[1]

    _, grad = compute_reward_gradient(xt, t, agent.classifier, fix_mask=fix_mask)
    x_shift = compute_optimization_shift(xt, grad, 0.1, fix_mask, prior)
    pred_shift = model["diffusion"](x_shift, t, None)

    pi_at_shift = compute_pi_t(x_shift, pred_shift, True, alpha, sigma)
    pi_at_xt = compute_pi_t(xt, pred_shift, True, alpha, sigma)

    assert not torch.allclose(pi_at_shift, pi_at_xt, atol=TOL)
    assert not torch.allclose(x_shift, xt, atol=TOL)


def test_optimization_eta_zero_matches_standard_ddim(planning_setup):
    agent, _ = planning_setup
    prior = _make_prior(2)

    torch.manual_seed(42)
    traj_std, _ = agent.sample(
        prior,
        solver="ddim",
        n_samples=2,
        sample_steps=5,
        use_ema=True,
        w_cg=0.0,
        guidance_mode="standard",
        optimization_guidance_scale=0.0,
        optimization_guidance_last_steps=5,
    )

    torch.manual_seed(42)
    traj_opt, _ = agent.sample(
        prior,
        solver="ddim",
        n_samples=2,
        sample_steps=5,
        use_ema=True,
        w_cg=0.0,
        guidance_mode="optimization",
        optimization_guidance_scale=0.0,
        optimization_guidance_last_steps=5,
    )

    assert torch.allclose(traj_std, traj_opt, atol=TOL)


def test_should_apply_optimization_guidance_requires_positive_eta():
    assert not should_apply_optimization_guidance(
        "optimization", loop_i=5, last_steps=10, optimization_guidance_scale=0.0
    )
    assert should_apply_optimization_guidance(
        "optimization", loop_i=5, last_steps=10, optimization_guidance_scale=0.1
    )
    assert should_apply_optimization_guidance(
        "hybrid", loop_i=5, last_steps=10, optimization_guidance_scale=0.1
    )


def test_hybrid_standard_guidance_only_before_last_steps():
    assert should_apply_standard_classifier_guidance(
        "hybrid", loop_i=15, last_steps=10, w_cg=0.3, use_optimization_guidance=False
    )
    assert not should_apply_standard_classifier_guidance(
        "hybrid", loop_i=5, last_steps=10, w_cg=0.3, use_optimization_guidance=True
    )


def test_fixed_observation_hard_enforced(planning_setup):
    agent, _ = planning_setup
    prior = _make_prior(3)

    traj, _ = agent.sample(
        prior,
        solver="ddpm",
        n_samples=3,
        sample_steps=5,
        use_ema=True,
        w_cg=0.0,
        guidance_mode="optimization",
        optimization_guidance_scale=0.05,
    )

    assert torch.allclose(traj[:, 0, :OBS_DIM], prior[:, 0, :OBS_DIM], atol=TOL)
