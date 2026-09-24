#!/usr/bin/env python3
"""Visualize Test A: reward contour, feasible square, and diffusion paths in subspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from cleandiffuser.classifier.runtime_reward import RuntimeRewardClassifier
from guidance_synthetic_subspace import (
    AMBIENT_DIM,
    HORIZON,
    GeometryMeta,
    _build_agent,
    _load_geometry,
    _make_reward_fn,
    _resolve_paths,
    _sample_batch_metrics,
    reward_values,
)
from cleandiffuser.utils import GaussianNormalizer
from utils import set_seed


def _load_normalizer(path: Path) -> GaussianNormalizer:
    normalizer = GaussianNormalizer(np.zeros((1, 1, AMBIENT_DIM)))
    npz = np.load(path)
    normalizer.mean = npz["mean"]
    normalizer.std = npz["std"]
    return normalizer


def ambient_to_subspace(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """x: (..., 10) -> (..., 2) subspace coords."""
    return x @ basis


def subspace_to_ambient(u: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """u: (..., 2) -> (..., 10)."""
    return u @ basis.T


def _extract_subspace_history(
    log: dict,
    normalizer: GaussianNormalizer,
    basis: np.ndarray,
) -> np.ndarray:
    """History for one sample: (steps+1, subspace_dim)."""
    hist = log["sample_history"]
    if hist.ndim == 5:
        x_norm = hist[0, :, 0, 0, :]
    elif hist.ndim == 4:
        x_norm = hist[0, :, 0, :]
    else:
        x_norm = hist[0]
    x_phys = normalizer.unnormalize(x_norm)
    return ambient_to_subspace(x_phys, basis)


def sample_paths_with_history(
    args,
    geometry: GeometryMeta,
    normalizer: GaussianNormalizer,
    agent,
    reward_fn,
    *,
    guidance_mode: str,
    w_cg: float,
    opt_scale: float,
    initial_xts: torch.Tensor,
) -> tuple[np.ndarray, dict]:
    """Sample one backward path per row of initial_xts; return (n_paths, steps+1, 2)."""
    if w_cg == 0.0 and opt_scale == 0.0:
        agent.classifier = None
    else:
        agent.classifier = RuntimeRewardClassifier(reward_fn, device=args.device)

    basis = geometry.basis_t
    n_paths = initial_xts.shape[0]
    prior = torch.zeros((1, HORIZON, AMBIENT_DIM), device=args.device)
    subspace_paths: list[np.ndarray] = []
    final_trajs: list[torch.Tensor] = []

    for i in range(n_paths):
        init = initial_xts[i : i + 1]
        traj, log = agent.sample(
            prior,
            solver=args.solver,
            n_samples=1,
            sample_steps=args.sampling_steps,
            sample_step_schedule=args.sample_step_schedule,
            use_ema=args.use_ema,
            w_cg=w_cg,
            guidance_mode=guidance_mode,
            optimization_guidance_scale=opt_scale,
            optimization_guidance_last_steps=args.optimization_guidance_last_steps,
            temperature=args.temperature,
            preserve_history=True,
            initial_xt=init,
        )
        subspace_paths.append(_extract_subspace_history(log, normalizer, basis))
        final_trajs.append(traj)

    paths = np.stack(subspace_paths, axis=0)
    metrics = _sample_batch_metrics(torch.cat(final_trajs, dim=0), geometry, normalizer)
    return paths, metrics


def _plot_backward_paths(
    ax,
    paths: np.ndarray,
    *,
    cmap_name: str,
    label_prefix: str,
    arrow_stride: int,
) -> None:
    n_paths, n_steps, _ = paths.shape
    cmap = plt.get_cmap(cmap_name, n_paths)
    for i in range(n_paths):
        color = cmap(i)
        p = paths[i]
        ax.plot(p[:, 0], p[:, 1], color=color, linewidth=1.4, alpha=0.85)
        ax.scatter(p[0, 0], p[0, 1], color=color, s=24, edgecolors="k", linewidths=0.4, zorder=6)
        ax.scatter(p[-1, 0], p[-1, 1], color=color, s=36, marker="x", linewidths=1.3, zorder=6)
        for t in range(0, n_steps - 1, arrow_stride):
            ax.annotate(
                "",
                xy=(p[t + 1, 0], p[t + 1, 1]),
                xytext=(p[t, 0], p[t, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=0.9, alpha=0.8),
            )
    ax.plot([], [], color=cmap(0.5), linewidth=2, label=f"{label_prefix} ({n_paths} inits × {n_steps - 1} steps)")


def plot_test_a(args) -> Path:
    set_seed(args.seed)
    root = Path(args.output_root) / "test_a"
    geometry = _load_geometry(root / "geometry.json")
    normalizer = _load_normalizer(root / "normalizer.npz")
    basis = geometry.basis_t
    target = geometry.target_t
    target_u = ambient_to_subspace(target[None, :], basis)[0]
    half_w = float(geometry.square_half_width)

    # --- reward contour on subspace grid ---
    lim = args.plot_limit
    n = args.grid_resolution
    u1 = np.linspace(-lim, lim, n)
    u2 = np.linspace(-lim, lim, n)
    U1, U2 = np.meshgrid(u1, u2)
    U = np.stack([U1, U2], axis=-1).reshape(-1, 2)
    X = subspace_to_ambient(U, basis)
    R = reward_values(X, geometry).reshape(n, n)

    # --- trajectories ---
    if args.w_cg is not None or args.opt_scale is not None:
        w_val = 0.0 if args.w_cg is None else args.w_cg
        o_val = 0.0 if args.opt_scale is None else args.opt_scale
        traj_specs = []
        if args.w_cg is not None:
            traj_specs.append((f"w_cg={w_val:g}", "standard", w_val, 0.0))
        if args.opt_scale is not None:
            traj_specs.append((f"opt={o_val:g}", "optimization", 0.0, o_val))
    elif args.traj_specs_json:
        traj_specs = [
            (s["label"], s["mode"], s["w_cg"], s["opt"])
            for s in json.loads(Path(args.traj_specs_json).read_text())
        ]
    else:
        grid_path = root / "sweep" / f"grid_s{args.sampling_steps}.json"
        if grid_path.is_file():
            grid = json.loads(grid_path.read_text())
            best_w = grid["best_w_cg"]
            best_o = grid["best_opt"]
            traj_specs = [
                ("unguided", "standard", 0.0, 0.0),
                (f"w_cg={best_w['w_cg']:g}", "standard", best_w["w_cg"], 0.0),
                (
                    f"opt={best_o['optimization_guidance_scale']:g}",
                    "optimization",
                    0.0,
                    best_o["optimization_guidance_scale"],
                ),
            ]
        else:
            traj_specs = [
                ("unguided", "standard", 0.0, 0.0),
                ("w_cg=5", "standard", 5.0, 0.0),
                ("opt=5", "optimization", 0.0, 5.0),
            ]

    ckpt_paths = _resolve_paths("test_a", Path(args.output_root))
    reward_fn = _make_reward_fn(geometry, normalizer, args.device)
    agent = _build_agent(args, classifier=None, diffusion_steps=args.diffusion_steps)
    agent.load(str(ckpt_paths["ckpt"]))
    agent.eval()

    # Shared initial noise at t=T for fair comparison across guidance configs.
    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    initial_xts = (
        torch.randn(
            (args.n_paths, HORIZON, AMBIENT_DIM),
            device=args.device,
            generator=gen,
        )
        * args.temperature
    )
    shared_starts = ambient_to_subspace(
        normalizer.unnormalize(initial_xts[:, 0, :].cpu().numpy()),
        basis,
    )

    metrics_report: list[dict] = []
    arrow_stride = max(1, args.sampling_steps // 10)

    fig, ax = plt.subplots(figsize=(9, 8))
    cf = ax.contourf(U1, U2, R, levels=40, cmap="viridis", alpha=0.85)
    plt.colorbar(cf, ax=ax, label="reward  $-(\\|x-x^*\\|^2)$")
    ax.contour(U1, U2, R, levels=20, colors="k", linewidths=0.3, alpha=0.35)

    # feasible square
    square = Rectangle(
        (-half_w, -half_w),
        2 * half_w,
        2 * half_w,
        facecolor="white",
        alpha=0.35,
        hatch="///",
        edgecolor="0.35",
        linewidth=1.5,
        label=f"feasible region ($\\pm${half_w:g} square)",
        zorder=2,
    )
    ax.add_patch(square)

    ax.scatter(
        [target_u[0]],
        [target_u[1]],
        c="red",
        s=120,
        marker="*",
        zorder=10,
        label=f"target $u^*$ ({target_u[0]:.2f}, {target_u[1]:.2f})",
    )

    for i, (u0, u1) in enumerate(shared_starts):
        ax.scatter(
            u0,
            u1,
            c="black",
            s=28,
            zorder=8,
            marker="o",
        )
        ax.text(u0 + 0.06, u1 + 0.06, str(i), fontsize=7, color="black", zorder=8)
    ax.plot([], [], color="black", marker="o", linestyle="None", label=f"shared init ($n$={args.n_paths})")

    cmap_by_mode = {"standard": "Oranges", "optimization": "PuBu"}
    for idx, (label, mode, w, opt) in enumerate(traj_specs):
        paths, metrics = sample_paths_with_history(
            args,
            geometry,
            normalizer,
            agent,
            reward_fn,
            guidance_mode=mode,
            w_cg=w,
            opt_scale=opt,
            initial_xts=initial_xts,
        )
        row = {
            "label": label,
            "guidance_mode": mode,
            "w_cg": w,
            "optimization_guidance_scale": opt,
            "n_paths": args.n_paths,
            "shared_initial_seed": args.seed,
            **metrics,
        }
        metrics_report.append(row)
        print(
            f"{label}: reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f}  "
            f"violation={metrics['violation_mean']:.4f}±{metrics['violation_std']:.4f}"
        )
        _plot_backward_paths(
            ax,
            paths,
            cmap_name=cmap_by_mode.get(mode, "viridis"),
            label_prefix=label,
            arrow_stride=arrow_stride,
        )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("subspace coord $u_1$")
    ax.set_ylabel("subspace coord $u_2$")
    ax.set_title(
        f"Test A: reward contour + {args.n_paths} backward diffusion paths "
        f"(DDIM {args.sampling_steps} steps, shared init)"
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        fig_name = args.output_name
    elif args.w_cg is not None and args.opt_scale is not None:
        fig_name = f"test_a_wcg{args.w_cg:g}_opt{args.opt_scale:g}_s{args.sampling_steps}"
    else:
        fig_name = f"test_a_subspace_contour_s{args.sampling_steps}"
    out_path = out_dir / f"{fig_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")

    metrics_path = out_dir / f"{fig_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"Wrote {metrics_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="results/guidance_synthetic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dim", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--training-diffusion-steps", type=int, default=50)
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--optimization-guidance-last-steps", type=int, default=25)
    p.add_argument("--predict-noise", action="store_true")
    p.add_argument("--ema-rate", type=float, default=0.9999)
    p.add_argument("--solver", default="ddim")
    p.add_argument("--sample-step-schedule", default="uniform")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--plot-limit", type=float, default=3.0)
    p.add_argument("--grid-resolution", type=int, default=200)
    p.add_argument("--n-paths", type=int, default=8)
    p.add_argument("--w-cg", type=float, default=None, help="Plot w_cg paths (standard mode)")
    p.add_argument("--opt-scale", type=float, default=None, help="Plot opt-scale paths (optimization mode)")
    p.add_argument("--output-name", default=None)
    p.add_argument("--traj-specs-json", default=None)
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def main():
    plot_test_a(parse_args())


if __name__ == "__main__":
    main()
