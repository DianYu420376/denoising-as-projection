#!/usr/bin/env python3
"""Paper-ready subspace figures: PDG baseline vs DCG guidance on Tests A and B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator
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
    constraint_violation,
    reward_values,
    solve_optimal_feasible,
)
from cleandiffuser.utils import GaussianNormalizer
from utils import set_seed

PDG_COLOR = "#2166AC"
DCG_COLOR = "#E66101"
FEASIBLE_FACE = "#EAF2FA"
FEASIBLE_FACE_ALPHA = 0.55
FEASIBLE_EDGE = "#2F4F6F"
FEASIBLE_EDGE_WIDTH = 1.8
INIT_COLOR = "#404040"
OPTIMAL_COLOR = "#C41E3A"

METHOD_STYLES = {
    "standard": {"color": PDG_COLOR, "label": "PDG Guidance (Baseline)"},
    "optimization": {"color": DCG_COLOR, "label": "DCG Guidance"},
}

# Per-test defaults from matched 50/50/50 sweeps.
TEST_DEFAULTS = {
    "test_a": {"w_cg": 0.6, "opt_scale": 15.0, "plot_limit": 2.0},
    "test_b": {"w_cg": 10.0, "opt_scale": 10.0, "plot_limit": 1.8},
}

PAPER_FONT_SCALE = 1.35
PATH_LEGEND_FONTSIZE = 9.0


def _apply_figure_style() -> None:
    s = PAPER_FONT_SCALE
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11 * s,
            "axes.labelsize": 12 * s,
            "axes.titlesize": 13 * s,
            "legend.fontsize": 10 * s,
            "xtick.labelsize": 10 * s,
            "ytick.labelsize": 10 * s,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#333333",
            "lines.linewidth": 2.2,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _style_axis_grid(ax, *, minor: bool = True) -> None:
    ax.grid(True, which="major", color="#D0D0D0", linewidth=0.9, alpha=0.95)
    if minor:
        ax.minorticks_on()
        ax.grid(True, which="minor", color="#E8E8E8", linewidth=0.55, alpha=0.95)


def _load_normalizer(path: Path) -> GaussianNormalizer:
    normalizer = GaussianNormalizer(np.zeros((1, 1, AMBIENT_DIM)))
    npz = np.load(path)
    normalizer.mean = npz["mean"]
    normalizer.std = npz["std"]
    return normalizer


def ambient_to_subspace(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return x @ basis


def subspace_to_ambient(u: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return u @ basis.T


def _test_params(args, test_name: str) -> tuple[float, float, float]:
    if test_name == "test_a":
        return args.w_cg, args.opt_scale, args.plot_limit_a
    return args.w_cg_test_b, args.opt_scale_test_b, args.plot_limit_b


def _slice_grid(geometry: GeometryMeta, lim: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u1 = np.linspace(-lim, lim, n)
    u2 = np.linspace(-lim, lim, n)
    u1g, u2g = np.meshgrid(u1, u2)
    sub_dim = geometry.subspace_dim
    u = np.zeros((n * n, sub_dim), dtype=np.float64)
    u[:, 0] = u1g.ravel()
    u[:, 1] = u2g.ravel()
    x = subspace_to_ambient(u, geometry.basis_t)
    rewards = reward_values(x, geometry).reshape(n, n)
    return u1g, u2g, rewards


def _extract_ambient_history(log: dict, normalizer: GaussianNormalizer) -> np.ndarray:
    """Return physical ambient trajectory with shape (steps+1, ambient_dim)."""
    hist = log["sample_history"]
    if hist.ndim == 5:
        x_norm = hist[0, :, 0, 0, :]
    elif hist.ndim == 4:
        x_norm = hist[0, :, 0, :]
    else:
        x_norm = hist[0]
    if x_norm.ndim == 3:
        x_norm = x_norm[:, 0, :]
    return normalizer.unnormalize(x_norm)


def _extract_subspace_history(
    log: dict,
    normalizer: GaussianNormalizer,
    basis: np.ndarray,
) -> np.ndarray:
    return ambient_to_subspace(_extract_ambient_history(log, normalizer), basis)


def _metrics_along_history(
    ambient_hist: np.ndarray,
    geometry: GeometryMeta,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = reward_values(ambient_hist, geometry)
    violations = constraint_violation(ambient_hist, geometry)
    return rewards, violations


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return subspace paths, reward curves, violation curves, and final metrics."""
    if w_cg == 0.0 and opt_scale == 0.0:
        agent.classifier = None
    else:
        agent.classifier = RuntimeRewardClassifier(reward_fn, device=args.device)

    n_paths = initial_xts.shape[0]
    prior = torch.zeros((1, HORIZON, AMBIENT_DIM), device=args.device)
    subspace_paths: list[np.ndarray] = []
    reward_curves: list[np.ndarray] = []
    violation_curves: list[np.ndarray] = []
    final_trajs: list[torch.Tensor] = []

    for i in range(n_paths):
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
            initial_xt=initial_xts[i : i + 1],
        )
        ambient_hist = _extract_ambient_history(log, normalizer)
        rewards, violations = _metrics_along_history(ambient_hist, geometry)
        subspace_paths.append(ambient_to_subspace(ambient_hist, geometry.basis_t))
        reward_curves.append(rewards)
        violation_curves.append(violations)
        final_trajs.append(traj)

    paths = np.stack(subspace_paths, axis=0)
    reward_curves = np.stack(reward_curves, axis=0)
    violation_curves = np.stack(violation_curves, axis=0)
    metrics = _sample_batch_metrics(torch.cat(final_trajs, dim=0), geometry, normalizer)
    return paths, reward_curves, violation_curves, metrics


def _plot_feasible_region(ax, geometry: GeometryMeta) -> None:
    if geometry.test == "test_a":
        half_w = float(geometry.square_half_width)
        patch = Rectangle(
            (-half_w, -half_w),
            2 * half_w,
            2 * half_w,
            facecolor=FEASIBLE_FACE,
            alpha=FEASIBLE_FACE_ALPHA,
            edgecolor=FEASIBLE_EDGE,
            linewidth=FEASIBLE_EDGE_WIDTH,
            linestyle="-",
            zorder=2,
            label="Feasible region",
        )
    else:
        axes = np.asarray(geometry.ellipsoid_axes, dtype=np.float64)
        patch = Ellipse(
            (0.0, 0.0),
            width=2.0 * axes[0],
            height=2.0 * axes[1],
            facecolor=FEASIBLE_FACE,
            alpha=FEASIBLE_FACE_ALPHA,
            edgecolor=FEASIBLE_EDGE,
            linewidth=FEASIBLE_EDGE_WIDTH,
            linestyle="-",
            zorder=2,
            label="Feasible region",
        )
    ax.add_patch(patch)


def _plot_backward_paths(
    ax,
    paths: np.ndarray,
    *,
    color: str,
    label: str,
    arrow_stride: int,
) -> None:
    n_paths, n_steps, _ = paths.shape
    for i in range(n_paths):
        p = paths[i]
        ax.plot(p[:, 0], p[:, 1], color=color, linewidth=1.4, alpha=0.78, zorder=5)
        ax.scatter(
            p[0, 0],
            p[0, 1],
            color=INIT_COLOR,
            s=16,
            edgecolors="white",
            linewidths=0.4,
            zorder=6,
        )
        ax.scatter(
            p[-1, 0],
            p[-1, 1],
            color=color,
            s=28,
            marker="x",
            linewidths=1.1,
            zorder=6,
        )
        for t in range(0, n_steps - 1, arrow_stride):
            ax.annotate(
                "",
                xy=(p[t + 1, 0], p[t + 1, 1]),
                xytext=(p[t, 0], p[t, 1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.7, alpha=0.55, shrinkA=0, shrinkB=0),
            )
    ax.plot([], [], color=color, linewidth=2.0, label=label)


def _save_figure(fig, out_dir: Path, fig_name: str) -> None:
    for ext in ("png", "pdf"):
        out_path = out_dir / f"{fig_name}.{ext}"
        fig.savefig(out_path, dpi=300 if ext == "png" else None)
        print(f"Wrote {out_path}")


def _trajectory_cache_paths(root: Path, fig_name: str) -> tuple[Path, Path]:
    cache_dir = root / "trajectories"
    return cache_dir / f"{fig_name}.npz", cache_dir / f"{fig_name}.meta.json"


def _sampling_metadata(args, test_name: str, w_cg: float, opt_scale: float) -> dict:
    return {
        "test": test_name,
        "seed": args.seed,
        "n_paths": args.n_paths,
        "w_cg": w_cg,
        "opt_scale": opt_scale,
        "diffusion_steps": args.diffusion_steps,
        "training_diffusion_steps": args.training_diffusion_steps,
        "sampling_steps": args.sampling_steps,
        "optimization_guidance_last_steps": args.optimization_guidance_last_steps,
        "solver": args.solver,
        "sample_step_schedule": args.sample_step_schedule,
        "temperature": args.temperature,
        "use_ema": args.use_ema,
        "predict_noise": args.predict_noise,
    }


def _save_trajectory_cache(
    cache_npz: Path,
    cache_meta: Path,
    run_data: dict,
    metadata: dict,
) -> None:
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for mode in ("standard", "optimization"):
        payload = run_data["methods"][mode]
        for key in ("paths", "reward_curves", "violation_curves"):
            arrays[f"{mode}_{key}"] = payload[key]
    arrays["feasible_optimum_subspace"] = np.asarray(
        run_data["feasible_optimum"]["subspace"], dtype=np.float64
    )
    arrays["feasible_optimum_ambient"] = np.asarray(
        run_data["feasible_optimum"]["ambient"], dtype=np.float64
    )
    np.savez_compressed(cache_npz, **arrays)

    meta_payload = {
        **metadata,
        "fig_name": run_data["fig_name"],
        "feasible_optimum": run_data["feasible_optimum"],
        "methods": {
            mode: {
                "label": run_data["methods"][mode]["label"],
                "metrics": run_data["methods"][mode]["metrics"],
            }
            for mode in ("standard", "optimization")
        },
    }
    with open(cache_meta, "w") as f:
        json.dump(meta_payload, f, indent=2)
    print(f"Wrote trajectory cache {cache_npz}")


def _load_trajectory_cache(cache_npz: Path, cache_meta: Path, metadata: dict) -> dict | None:
    if not cache_npz.is_file() or not cache_meta.is_file():
        return None
    with open(cache_meta) as f:
        cached = json.load(f)
    for key, value in metadata.items():
        if cached.get(key) != value:
            return None

    data = np.load(cache_npz)
    run_data = {
        "test": cached["test"],
        "w_cg": cached["w_cg"],
        "opt_scale": cached["opt_scale"],
        "fig_name": cached["fig_name"],
        "feasible_optimum": cached["feasible_optimum"],
        "methods": {},
    }
    for mode in ("standard", "optimization"):
        run_data["methods"][mode] = {
            "label": METHOD_STYLES[mode]["label"],
            "paths": data[f"{mode}_paths"],
            "reward_curves": data[f"{mode}_reward_curves"],
            "violation_curves": data[f"{mode}_violation_curves"],
            "metrics": cached["methods"][mode]["metrics"],
        }
    print(f"Loaded trajectory cache {cache_npz}")
    return run_data


def _sample_run_data(
    args,
    test_name: str,
    geometry: GeometryMeta,
    normalizer: GaussianNormalizer,
    *,
    w_cg: float,
    opt_scale: float,
    fig_name: str,
) -> dict:
    traj_specs = [
        ("standard", w_cg, 0.0),
        ("optimization", 0.0, opt_scale),
    ]

    ckpt_paths = _resolve_paths(test_name, Path(args.output_root))
    reward_fn = _make_reward_fn(geometry, normalizer, args.device)
    agent = _build_agent(args, classifier=None, diffusion_steps=args.diffusion_steps)
    agent.load(str(ckpt_paths["ckpt"]))
    agent.eval()

    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    initial_xts = (
        torch.randn((args.n_paths, HORIZON, AMBIENT_DIM), device=args.device, generator=gen)
        * args.temperature
    )

    x_opt, u_opt = solve_optimal_feasible(geometry)
    opt_reward = float(reward_values(x_opt[None], geometry)[0])
    opt_violation = float(constraint_violation(x_opt[None], geometry)[0])
    run_data: dict = {
        "test": test_name,
        "w_cg": w_cg,
        "opt_scale": opt_scale,
        "fig_name": fig_name,
        "feasible_optimum": {
            "ambient": x_opt.tolist(),
            "subspace": u_opt.tolist(),
            "reward": opt_reward,
            "violation": opt_violation,
        },
        "methods": {},
    }
    print(
        f"[{test_name}] Feasible optimum: u={u_opt[:2]}, "
        f"reward={opt_reward:.4f}, violation={opt_violation:.6f}"
    )

    for mode, w_val, opt_val in traj_specs:
        paths, reward_curves, violation_curves, metrics = sample_paths_with_history(
            args,
            geometry,
            normalizer,
            agent,
            reward_fn,
            guidance_mode=mode,
            w_cg=w_val,
            opt_scale=opt_val,
            initial_xts=initial_xts,
        )
        style = METHOD_STYLES[mode]
        run_data["methods"][mode] = {
            "label": style["label"],
            "paths": paths,
            "reward_curves": reward_curves,
            "violation_curves": violation_curves,
            "metrics": metrics,
        }
        print(
            f"[{test_name}] {style['label']} (w_cg={w_val:g}, opt={opt_val:g}): "
            f"reward={metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f}  "
            f"violation={metrics['violation_mean']:.4f}±{metrics['violation_std']:.4f}"
        )
    return run_data


def _render_paths_figure(
    args,
    test_name: str,
    geometry: GeometryMeta,
    plot_limit: float,
    run_data: dict,
) -> None:
    u1g, u2g, rewards = _slice_grid(geometry, plot_limit, args.grid_resolution)
    arrow_stride = max(1, args.sampling_steps // 10)

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    levels = np.linspace(rewards.min(), min(0.0, rewards.max()), 20)
    cf = ax.contourf(u1g, u2g, rewards, levels=levels, cmap="bone", alpha=0.55, zorder=1)
    ax.contour(u1g, u2g, rewards, levels=levels[::2], colors="#BDBDBD", linewidths=0.5, alpha=0.8, zorder=1)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=PATH_LEGEND_FONTSIZE)

    _plot_feasible_region(ax, geometry)
    u_opt = np.asarray(run_data["feasible_optimum"]["subspace"], dtype=np.float64)
    ax.scatter(
        u_opt[0],
        u_opt[1],
        marker="*",
        s=160,
        color=OPTIMAL_COLOR,
        edgecolors="white",
        linewidths=0.7,
        zorder=7,
        label="Optimal (feasible)",
    )

    for mode in ("standard", "optimization"):
        style = METHOD_STYLES[mode]
        _plot_backward_paths(
            ax,
            run_data["methods"][mode]["paths"],
            color=style["color"],
            label=style["label"],
            arrow_stride=arrow_stride,
        )

    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        loc="upper left",
        fontsize=PATH_LEGEND_FONTSIZE,
        markerscale=0.8,
        handlelength=1.5,
        borderpad=0.35,
        labelspacing=0.3,
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
    )
    _style_axis_grid(ax, minor=True)

    out_dir = Path(args.output_root) / test_name / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, out_dir, run_data["fig_name"])
    plt.close(fig)


def load_run_data(args, test_name: str) -> dict:
    """Load cached trajectories or sample if needed."""
    w_cg, opt_scale, _plot_limit = _test_params(args, test_name)
    root = Path(args.output_root) / test_name
    geometry = _load_geometry(root / "geometry.json")
    normalizer = _load_normalizer(root / "normalizer.npz")

    fig_name = args.output_name or (
        f"{test_name}_pcg_dcg_fig_s{args.sampling_steps}_last{args.optimization_guidance_last_steps}"
    )
    metadata = _sampling_metadata(args, test_name, w_cg, opt_scale)
    cache_npz, cache_meta = _trajectory_cache_paths(root, fig_name)

    run_data = None
    if not args.resample:
        run_data = _load_trajectory_cache(cache_npz, cache_meta, metadata)

    if run_data is None:
        if args.plot_only or args.convergence_only or args.paths_only:
            raise FileNotFoundError(
                f"No trajectory cache for {test_name} at {cache_npz}. "
                "Run without --plot-only/--convergence-only first (or pass --resample)."
            )
        run_data = _sample_run_data(
            args,
            test_name,
            geometry,
            normalizer,
            w_cg=w_cg,
            opt_scale=opt_scale,
            fig_name=fig_name,
        )
        _save_trajectory_cache(cache_npz, cache_meta, run_data, metadata)
    return run_data


def plot_paths(args, test_name: str) -> dict:
    _apply_figure_style()
    set_seed(args.seed)

    w_cg, opt_scale, plot_limit = _test_params(args, test_name)
    root = Path(args.output_root) / test_name
    geometry = _load_geometry(root / "geometry.json")

    run_data = load_run_data(args, test_name)
    if not args.convergence_only:
        _render_paths_figure(args, test_name, geometry, plot_limit, run_data)
    return run_data


def _violation_tick_values(ymin: float, ymax: float) -> list[float]:
    ticks: list[float] = []
    p_lo = int(np.floor(np.log10(ymin))) - 1
    p_hi = int(np.ceil(np.log10(ymax))) + 1
    for p in range(p_lo, p_hi + 1):
        for mult in (1.0, 2.0, 5.0):
            val = mult * (10.0**p)
            if ymin <= val <= ymax:
                ticks.append(val)
    return ticks


def _violation_tick_label(value: float, _pos: int) -> str:
    exp = int(np.floor(np.log10(value)))
    coeff = value / (10.0**exp)
    if abs(coeff - 1.0) < 1e-6:
        return rf"$10^{{{exp}}}$"
    return rf"${coeff:g}\times10^{{{exp}}}$"


def _configure_violation_axis(ax) -> None:
    """Log-scale violation axis: keep autoscale y-limits, 1-2-5 ticks inside view."""
    ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    ticks = _violation_tick_values(ymin, ymax)
    if ticks:
        ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_violation_tick_label))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.set_ylim(ymin, ymax)


def plot_convergence(args, test_name: str, run_data: dict) -> None:
    _apply_figure_style()
    steps = np.arange(run_data["methods"]["standard"]["reward_curves"].shape[1])
    opt_objective = -float(run_data["feasible_optimum"]["reward"])
    viol_floor = 1e-4

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    panels = [
        ("reward_curves", "Objective", axes[0]),
        ("violation_curves", "Constraint violation", axes[1]),
    ]

    for key, subtitle, ax in panels:
        for mode in ("standard", "optimization"):
            curves = run_data["methods"][mode][key]
            style = METHOD_STYLES[mode]
            if key == "reward_curves":
                curves = -curves
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            if key == "violation_curves":
                mean = np.maximum(mean, viol_floor)
                mean_lo = np.maximum(mean - std, viol_floor)
                mean_hi = mean + std
            else:
                mean_lo = mean - std
                mean_hi = mean + std
            ax.plot(steps, mean, color=style["color"], linewidth=2.6, label=style["label"])
            ax.fill_between(steps, mean_lo, mean_hi, color=style["color"], alpha=0.18)
        if key == "reward_curves":
            ax.axhline(
                opt_objective,
                color="black",
                linestyle="--",
                linewidth=2.0,
                label="Optimal (feasible)",
                zorder=3,
            )
        if key == "violation_curves":
            _configure_violation_axis(ax)
        ax.set_xlabel("Backward diffusion step")
        ax.set_title(subtitle)
        _style_axis_grid(ax, minor=True)

    axes[0].legend(loc="best", frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
    fig.tight_layout()

    out_dir = Path(args.output_root) / test_name / "figures"
    conv_name = f"{run_data['fig_name']}_convergence"
    _save_figure(fig, out_dir, conv_name)
    plt.close(fig)


def plot_training_data_sanity(args, test_name: str) -> None:
    """Scatter training data in (u1, u2) subspace coords with feasible region overlay."""
    _apply_figure_style()
    root = Path(args.output_root) / test_name
    geometry = _load_geometry(root / "geometry.json")
    dataset_path = root / "dataset.npz"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing training dataset: {dataset_path}")

    data = np.load(dataset_path)["data"]
    basis = geometry.basis_t
    u = ambient_to_subspace(data.astype(np.float64), basis)

    if test_name == "test_a":
        plot_limit = args.plot_limit_a
    else:
        plot_limit = args.plot_limit_b

    max_pts = args.sanity_max_points
    if len(u) > max_pts:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(u), size=max_pts, replace=False)
        u_plot = u[idx]
    else:
        u_plot = u

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    _plot_feasible_region(ax, geometry)
    ax.scatter(
        u_plot[:, 0],
        u_plot[:, 1],
        s=4,
        alpha=0.25,
        color="#4D4D4D",
        edgecolors="none",
        rasterized=True,
        label=f"Training data ({len(u_plot):,} shown)",
        zorder=3,
    )

    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Subspace coordinate $u_1$")
    ax.set_ylabel(r"Subspace coordinate $u_2$")
    title = "Test A" if test_name == "test_a" else "Test B"
    ax.set_title(f"{title}: training distribution vs feasible region")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
    ax.grid(True, color="#E6E6E6", linewidth=0.6, alpha=0.8)

    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_name = f"{test_name}_training_data_sanity"
    _save_figure(fig, out_dir, fig_name)
    plt.close(fig)
    print(f"[{test_name}] training samples={len(data):,}, plotted={len(u_plot):,}")


def _write_metrics(args, test_name: str, run_data: dict) -> None:
    metrics_report = []
    for mode, payload in run_data["methods"].items():
        metrics_report.append(
            {
                "test": test_name,
                "label": payload["label"],
                "guidance_mode": mode,
                "w_cg": run_data["w_cg"] if mode == "standard" else 0.0,
                "optimization_guidance_scale": run_data["opt_scale"] if mode == "optimization" else 0.0,
                "n_paths": args.n_paths,
                "shared_initial_seed": args.seed,
                **payload["metrics"],
            }
        )
    out_path = Path(args.output_root) / test_name / "figures" / f"{run_data['fig_name']}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "feasible_optimum": run_data.get("feasible_optimum"),
                "methods": metrics_report,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["test_a", "test_b", "both"], default="both")
    p.add_argument("--output-root", default="results/guidance_synthetic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dim", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--training-diffusion-steps", type=int, default=50)
    p.add_argument("--sampling-steps", type=int, default=50)
    p.add_argument("--optimization-guidance-last-steps", type=int, default=50)
    p.add_argument("--predict-noise", action="store_true")
    p.add_argument("--ema-rate", type=float, default=0.9999)
    p.add_argument("--solver", default="ddim")
    p.add_argument("--sample-step-schedule", default="uniform")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--plot-limit-a", type=float, default=TEST_DEFAULTS["test_a"]["plot_limit"])
    p.add_argument("--plot-limit-b", type=float, default=TEST_DEFAULTS["test_b"]["plot_limit"])
    p.add_argument("--grid-resolution", type=int, default=200)
    p.add_argument("--n-paths", type=int, default=5)
    p.add_argument("--w-cg", type=float, default=TEST_DEFAULTS["test_a"]["w_cg"])
    p.add_argument("--opt-scale", type=float, default=TEST_DEFAULTS["test_a"]["opt_scale"])
    p.add_argument("--w-cg-test-b", type=float, default=TEST_DEFAULTS["test_b"]["w_cg"])
    p.add_argument("--opt-scale-test-b", type=float, default=TEST_DEFAULTS["test_b"]["opt_scale"])
    p.add_argument("--output-name", default=None)
    p.add_argument("--skip-convergence", action="store_true")
    p.add_argument(
        "--resample",
        action="store_true",
        help="Force re-sampling and overwrite saved trajectory cache.",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot from saved trajectories only (skip GPU sampling).",
    )
    p.add_argument(
        "--convergence-only",
        action="store_true",
        help="Replot convergence panels from cached trajectories only (fastest).",
    )
    p.add_argument(
        "--paths-only",
        action="store_true",
        help="Replot path figures from cached trajectories only.",
    )
    p.add_argument(
        "--sanity-data",
        action="store_true",
        help="Only plot training data vs feasible region (no GPU sampling).",
    )
    p.add_argument("--sanity-max-points", type=int, default=20_000)
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def main():
    args = parse_args()
    tests = ["test_a", "test_b"] if args.test == "both" else [args.test]

    if args.sanity_data:
        for test_name in tests:
            plot_training_data_sanity(args, test_name)
        return

    for test_name in tests:
        run_data = plot_paths(args, test_name)
        if not args.skip_convergence and not args.paths_only:
            plot_convergence(args, test_name, run_data)
        if not args.convergence_only and not args.paths_only:
            _write_metrics(args, test_name, run_data)


if __name__ == "__main__":
    main()
