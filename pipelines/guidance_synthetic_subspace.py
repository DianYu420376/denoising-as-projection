#!/usr/bin/env python3
"""Synthetic subspace guidance tests (non-D4RL).

Test A: uniform data on a 2D linear subspace inside a square (ambient R^10).
Test B: uniform data on a 5D linear subspace inside an ellipsoid (ambient R^10).

Both use a convex reward whose maximizer lies outside the feasible support of the data.
Modes:
  train     — build dataset, pretrain diffusion on the data distribution
  inference — sample with guidance; log reward and constraint violation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from cleandiffuser.classifier.runtime_reward import RuntimeRewardClassifier
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.diffusion.guidance import validate_guidance_config
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import GaussianNormalizer, report_parameters
from guidance_comparison_eval import build_configs
from utils import set_seed

AMBIENT_DIM = 10
HORIZON = 1


@dataclass
class GeometryMeta:
    test: str
    ambient_dim: int
    subspace_dim: int
    basis: list[list[float]]
    target: list[float]
    square_half_width: float | None = None
    ellipsoid_axes: list[float] | None = None

    @property
    def basis_t(self) -> np.ndarray:
        return np.asarray(self.basis, dtype=np.float64)

    @property
    def target_t(self) -> np.ndarray:
        return np.asarray(self.target, dtype=np.float64)


def _orthonormal_basis(ambient_dim: int, subspace_dim: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal((ambient_dim, subspace_dim))
    q, _ = np.linalg.qr(raw)
    return q.astype(np.float64)


def _sample_uniform_ellipsoid(
    axes: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    """Uniform samples in {u : sum_i (u_i / a_i)^2 <= 1} via rejection."""
    axes = np.asarray(axes, dtype=np.float64)
    sub_dim = axes.shape[0]
    samples: list[np.ndarray] = []
    while sum(len(s) for s in samples) < n:
        cand = rng.uniform(-1.0, 1.0, size=(batch_size, sub_dim)) * axes[None, :]
        norm_sq = np.sum((cand / axes[None, :]) ** 2, axis=1)
        accepted = cand[norm_sq <= 1.0 + 1e-12]
        if len(accepted) > 0:
            samples.append(accepted)
    return np.vstack(samples)[:n]


def build_test_a_geometry(seed: int, *, half_width: float = 1.0, target_scale: float = 2.5) -> GeometryMeta:
    rng = np.random.default_rng(seed)
    basis = _orthonormal_basis(AMBIENT_DIM, 2, rng)
    target_sub = np.array([target_scale * half_width, target_scale * half_width], dtype=np.float64)
    target = basis @ target_sub
    return GeometryMeta(
        test="test_a",
        ambient_dim=AMBIENT_DIM,
        subspace_dim=2,
        basis=basis.tolist(),
        target=target.tolist(),
        square_half_width=float(half_width),
    )


def build_test_b_geometry(
    seed: int,
    *,
    axes: tuple[float, ...] = (1.0, 0.85, 0.7, 0.55, 0.4),
    target_scale: float = 2.2,
) -> GeometryMeta:
    rng = np.random.default_rng(seed)
    basis = _orthonormal_basis(AMBIENT_DIM, 5, rng)
    axes_arr = np.asarray(axes, dtype=np.float64)
    target_sub = target_scale * axes_arr
    target = basis @ target_sub
    return GeometryMeta(
        test="test_b",
        ambient_dim=AMBIENT_DIM,
        subspace_dim=5,
        basis=basis.tolist(),
        target=target.tolist(),
        ellipsoid_axes=axes_arr.tolist(),
    )


def sample_test_a(n: int, geometry: GeometryMeta, rng: np.random.Generator) -> np.ndarray:
    w = float(geometry.square_half_width)
    u = rng.uniform(-w, w, size=(n, 2))
    return (u @ geometry.basis_t.T).astype(np.float32)


def sample_test_b(n: int, geometry: GeometryMeta, rng: np.random.Generator) -> np.ndarray:
    axes = np.asarray(geometry.ellipsoid_axes, dtype=np.float64)
    u = _sample_uniform_ellipsoid(axes, n, rng)
    return (u @ geometry.basis_t.T).astype(np.float32)


def subspace_coords(x: np.ndarray, geometry: GeometryMeta) -> np.ndarray:
    return x @ geometry.basis_t


def project_to_feasible(x: np.ndarray, geometry: GeometryMeta) -> np.ndarray:
    u = subspace_coords(x, geometry)
    if geometry.test == "test_a":
        w = float(geometry.square_half_width)
        u_proj = np.clip(u, -w, w)
    else:
        axes = np.asarray(geometry.ellipsoid_axes, dtype=np.float64)
        norm_sq = np.sum((u / axes[None, :]) ** 2, axis=1, keepdims=True)
        scale = np.where(norm_sq > 1.0, 1.0 / np.sqrt(norm_sq), 1.0)
        u_proj = u * scale
    return u_proj @ geometry.basis_t.T


def constraint_violation(x: np.ndarray, geometry: GeometryMeta) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    proj = project_to_feasible(x, geometry)
    return np.linalg.norm(x - proj, axis=1)


def reward_values(x: np.ndarray, geometry: GeometryMeta) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    diff = x - geometry.target_t[None, :]
    return -(diff ** 2).sum(axis=1)


def solve_optimal_feasible(geometry: GeometryMeta) -> tuple[np.ndarray, np.ndarray]:
    """Solve for x* in feasible set that maximizes R(x) = -||x - target||^2.

    Returns (ambient_point, subspace_coords).
    """
    from scipy.optimize import minimize

    basis = geometry.basis_t
    target = geometry.target_t
    sub_dim = geometry.subspace_dim
    u0 = subspace_coords(target[None], geometry).reshape(sub_dim)

    def objective(u: np.ndarray) -> float:
        x = u @ basis.T
        return float(np.sum((x - target) ** 2))

    if geometry.test == "test_a":
        w = float(geometry.square_half_width)
        bounds = [(-w, w)] * sub_dim
        result = minimize(objective, u0, method="L-BFGS-B", bounds=bounds)
    elif geometry.test == "test_b":
        axes = np.asarray(geometry.ellipsoid_axes, dtype=np.float64)

        def ellipsoid_margin(u: np.ndarray) -> float:
            return 1.0 - float(np.sum((u / axes) ** 2))

        result = minimize(
            objective,
            u0 * 0.5,
            method="SLSQP",
            constraints={"type": "ineq", "fun": ellipsoid_margin},
        )
    else:
        raise ValueError(f"Unknown test {geometry.test!r}")

    if not result.success:
        raise RuntimeError(f"Feasible optimum solve failed: {result.message}")

    u_opt = np.asarray(result.x, dtype=np.float64)
    x_opt = u_opt @ basis.T
    return x_opt, u_opt


def _resolve_paths(test: str, output_root: Path) -> dict[str, Path]:
    root = output_root / test
    return {
        "root": root,
        "dataset": root / "dataset.npz",
        "geometry": root / "geometry.json",
        "ckpt": root / "diffusion_ckpt_latest.pt",
        "normalizer": root / "normalizer.npz",
    }


def _remove_existing_artifacts(paths: dict[str, Path]) -> None:
    for key in ("dataset", "geometry", "normalizer", "ckpt"):
        path = paths[key]
        if path.is_file():
            path.unlink()
            print(f"[recreate] removed {path}")


def _save_geometry(path: Path, geometry: GeometryMeta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(geometry), f, indent=2)


def _load_geometry(path: Path) -> GeometryMeta:
    with open(path) as f:
        payload = json.load(f)
    return GeometryMeta(**payload)


def create_dataset(args) -> tuple[np.ndarray, GeometryMeta, GaussianNormalizer]:
    paths = _resolve_paths(args.test, Path(args.output_root))
    if getattr(args, "recreate_dataset", False):
        _remove_existing_artifacts(paths)
    if args.test == "test_a":
        geometry = build_test_a_geometry(args.seed, half_width=args.square_half_width)
        sampler = sample_test_a
    elif args.test == "test_b":
        geometry = build_test_b_geometry(args.seed, axes=tuple(args.ellipsoid_axes))
        sampler = sample_test_b
    else:
        raise ValueError(f"Unknown test {args.test!r}")

    rng = np.random.default_rng(args.seed + 17)
    data = sampler(args.num_dataset_samples, geometry, rng)
    normalizer = GaussianNormalizer(data[:, None, :], start_dim=-1)

    paths["root"].mkdir(parents=True, exist_ok=True)
    np.savez(paths["dataset"], data=data)
    np.savez(paths["normalizer"], mean=normalizer.mean, std=normalizer.std)
    _save_geometry(paths["geometry"], geometry)

    print(
        f"[dataset] {args.test}: n={len(data)} ambient={AMBIENT_DIM} subspace={geometry.subspace_dim} "
        f"target_norm={np.linalg.norm(geometry.target_t):.3f} "
        f"mean_viol(target)={constraint_violation(geometry.target_t[None], geometry)[0]:.3f}"
    )
    return data, geometry, normalizer


def _load_dataset_bundle(args) -> tuple[np.ndarray, GeometryMeta, GaussianNormalizer]:
    paths = _resolve_paths(args.test, Path(args.output_root))
    if not paths["dataset"].is_file():
        return create_dataset(args)
    data = np.load(paths["dataset"])["data"]
    normalizer = GaussianNormalizer(np.zeros((1, 1, AMBIENT_DIM)))
    norm_npz = np.load(paths["normalizer"])
    normalizer.mean = norm_npz["mean"]
    normalizer.std = norm_npz["std"]
    geometry = _load_geometry(paths["geometry"])
    return data, geometry, normalizer


def _build_agent(
    args,
    classifier: RuntimeRewardClassifier | None = None,
    *,
    diffusion_steps: int | None = None,
) -> DiscreteDiffusionSDE:
    nn_diffusion = JannerUNet1d(
        AMBIENT_DIM,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=[1],
        timestep_emb_type="positional",
        attention=False,
        kernel_size=3,
    )
    fix_mask = torch.zeros((HORIZON, AMBIENT_DIM))
    loss_weight = torch.ones((HORIZON, AMBIENT_DIM))
    steps = args.diffusion_steps if diffusion_steps is None else diffusion_steps
    agent = DiscreteDiffusionSDE(
        nn_diffusion,
        None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=classifier,
        ema_rate=args.ema_rate,
        device=args.device,
        diffusion_steps=steps,
        training_diffusion_steps=args.training_diffusion_steps,
        predict_noise=args.predict_noise,
    )
    return agent


def train(args) -> None:
    set_seed(args.seed)
    paths = _resolve_paths(args.test, Path(args.output_root))
    if getattr(args, "recreate_dataset", False):
        data, geometry, normalizer = create_dataset(args)
    else:
        data, geometry, normalizer = _load_dataset_bundle(args)

    norm_data = normalizer.normalize(data[:, None, :]).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(norm_data)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    agent = _build_agent(args, classifier=None)
    print("======================= Diffusion Parameter Report =======================")
    report_parameters(agent.model)
    print("==========================================================================")

    agent.train()
    scheduler = CosineAnnealingLR(agent.optimizer, args.train_steps)
    step = 0
    log_loss = 0.0

    while step < args.train_steps:
        for (batch,) in loader:
            loss = agent.update(batch.to(args.device))["loss"]
            log_loss += float(loss)
            scheduler.step()
            step += 1

            if step % args.log_interval == 0:
                print(
                    {
                        "step": step,
                        "avg_loss": log_loss / args.log_interval,
                        "test": args.test,
                    }
                )
                log_loss = 0.0

            if step % args.save_interval == 0 or step == args.train_steps:
                agent.save(str(paths["ckpt"]))

            if step >= args.train_steps:
                break

    print(f"[train] saved checkpoint to {paths['ckpt']}")


def _make_reward_fn(geometry: GeometryMeta, normalizer: GaussianNormalizer, device: str):
    mean = torch.tensor(normalizer.mean, device=device, dtype=torch.float32).reshape(1, 1, -1)
    std = torch.tensor(normalizer.std, device=device, dtype=torch.float32).reshape(1, 1, -1)
    target = torch.tensor(geometry.target_t, device=device, dtype=torch.float32)

    def reward_fn(x_norm: torch.Tensor, c=None) -> torch.Tensor:
        x_phys = x_norm * std + mean
        x_flat = x_phys.reshape(x_phys.shape[0], AMBIENT_DIM)
        return -((x_flat - target) ** 2).sum(dim=1)

    return reward_fn


def _sample_batch_metrics(
    x_norm: torch.Tensor,
    geometry: GeometryMeta,
    normalizer: GaussianNormalizer,
) -> dict[str, float]:
    x_np = normalizer.unnormalize(x_norm.detach().cpu().numpy()).reshape(-1, AMBIENT_DIM)
    rewards = reward_values(x_np, geometry)
    violations = constraint_violation(x_np, geometry)
    return {
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "violation_mean": float(violations.mean()),
        "violation_std": float(violations.std()),
        "violation_max": float(violations.max()),
    }


def inference(args) -> None:
    set_seed(args.seed)
    paths = _resolve_paths(args.test, Path(args.output_root))
    _, geometry, normalizer = _load_dataset_bundle(args)

    configs = build_configs(
        args.opt_scales,
        include_monte_carlo=False,
        config_names=args.config_names,
    )

    out_root = paths["root"] / "inference"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for cfg in configs:
        validate_guidance_config(cfg["guidance_mode"], cfg["w_cg"], cfg["optimization_guidance_scale"])
        reward_fn = _make_reward_fn(geometry, normalizer, args.device)
        diff_steps = args.diffusion_steps
        agent = _build_agent(
            args,
            classifier=RuntimeRewardClassifier(reward_fn, device=args.device),
            diffusion_steps=diff_steps,
        )
        if cfg["w_cg"] == 0.0 and cfg["optimization_guidance_scale"] == 0.0:
            agent.classifier = None
        agent.load(str(paths["ckpt"]))
        agent.eval()

        prior = torch.zeros((args.num_samples, HORIZON, AMBIENT_DIM), device=args.device)
        traj, _ = agent.sample(
            prior,
            solver=args.solver,
            n_samples=args.num_samples,
            sample_steps=args.sampling_steps,
            sample_step_schedule=args.sample_step_schedule,
            use_ema=args.use_ema,
            w_cg=cfg["w_cg"],
            guidance_mode=cfg["guidance_mode"],
            optimization_guidance_scale=cfg["optimization_guidance_scale"],
            optimization_guidance_last_steps=args.optimization_guidance_last_steps,
            temperature=args.temperature,
        )

        metrics = _sample_batch_metrics(traj, geometry, normalizer)
        row = {
            "test": args.test,
            "config": cfg["name"],
            "guidance_mode": cfg["guidance_mode"],
            "w_cg": cfg["w_cg"],
            "optimization_guidance_scale": cfg["optimization_guidance_scale"],
            "num_samples": args.num_samples,
            **metrics,
        }
        summary_rows.append(row)

        cfg_dir = out_root / cfg["name"]
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_dir / "summary.json", "w") as f:
            json.dump(row, f, indent=2)

        print(
            f"[inference] {args.test} {cfg['name']}: "
            f"reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f} "
            f"violation={metrics['violation_mean']:.4f}±{metrics['violation_std']:.4f}"
        )

    summary_path = out_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"runs": summary_rows}, f, indent=2)
    print(f"[inference] wrote {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic subspace diffusion guidance tests A & B")
    parser.add_argument("--mode", choices=["train", "inference", "create_dataset"], required=True)
    parser.add_argument("--test", choices=["test_a", "test_b"], required=True)
    parser.add_argument("--output-root", default="results/guidance_synthetic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--num-dataset-samples", type=int, default=200_000)
    parser.add_argument(
        "--recreate-dataset",
        action="store_true",
        help="Delete and regenerate dataset/geometry/normalizer (and ckpt on train).",
    )
    parser.add_argument("--square-half-width", type=float, default=1.0)
    parser.add_argument(
        "--ellipsoid-axes",
        nargs=5,
        type=float,
        default=[1.0, 0.85, 0.7, 0.55, 0.4],
    )

    parser.add_argument("--train-steps", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=20_000)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--training-diffusion-steps", type=int, default=50)
    parser.add_argument("--predict-noise", action="store_true")
    parser.add_argument("--ema-rate", type=float, default=0.9999)

    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--solver", default="ddim")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--sample-step-schedule", default="uniform")
    parser.add_argument(
        "--inference-diffusion-steps",
        type=int,
        default=None,
        help="Deprecated: kept for compatibility; use --diffusion-steps (all three should match).",
    )
    parser.add_argument("--optimization-guidance-last-steps", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--opt-scales", nargs="+", type=float, default=[30.0, 40.0, 50.0, 60.0])
    parser.add_argument(
        "--config-names",
        nargs="+",
        default=[
            "standard_w_cg0",
            "standard_w_cg10",
            "standard_w_cg20",
            "standard_w_cg30",
            "optimization_scale_30",
            "optimization_scale_40",
            "optimization_scale_50",
            "optimization_scale_60",
        ],
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def main():
    args = parse_args()
    if args.mode == "create_dataset":
        create_dataset(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "inference":
        inference(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
