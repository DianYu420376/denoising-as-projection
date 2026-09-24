#!/usr/bin/env python3
"""Fast replot from candidates.npz — no torch/gym/diffusion imports."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLAN_PLOT_LEGEND_FONTSIZE = 14
PLAN_PLOT_DPI = 200

PLOT_STYLE_DEFAULT = {
    "line_width": 2.0,
    "ref_line_width": 2.2,
    "start_size": 110.0,
}
PLOT_STYLE_EMPHASIS = {
    "line_width": 3.7,
    "ref_line_width": 3.9,
    "start_size": 660.0,
}
EMPHASIS_TRAJECTORIES = {"circle", "random_smooth_1", "random_smooth_7"}
EMPHASIS_CONFIGS = {"standard_w_cg30", "optimization_scale_30"}


def resolve_plot_style(trajectory: str, config: str) -> dict:
    if trajectory in EMPHASIS_TRAJECTORIES and config in EMPHASIS_CONFIGS:
        return PLOT_STYLE_EMPHASIS
    return PLOT_STYLE_DEFAULT


def apply_figure_plot_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.default": "regular",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
            "font.size": 12,
            "axes.labelsize": 14,
            "legend.fontsize": PLAN_PLOT_LEGEND_FONTSIZE,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.0,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def centered_xy_limits(*xy_arrays: np.ndarray, pad_frac: float = 0.14, min_span: float = 1.5):
    chunks = [np.asarray(arr, dtype=np.float64).reshape(-1, 2) for arr in xy_arrays if arr is not None]
    pts = np.vstack(chunks)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(float(xmax - xmin), float(ymax - ymin), min_span)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * span * (1.0 + pad_frac)
    return (cx - half, cx + half), (cy - half, cy + half)


def setup_plan_axes(ax, x_lim, y_lim) -> None:
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, which="major", alpha=0.45)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.22, linestyle=":")


def save_plan_figure(fig, output_path: Path, *, save_pdf: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLAN_PLOT_DPI)
    if save_pdf and output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def save_plan_legend_figure(
    output_path: Path,
    *,
    save_pdf: bool = False,
    style: dict | None = None,
) -> None:
    """Standalone wide legend bar for grouped figures."""
    from matplotlib.lines import Line2D

    style = style or PLOT_STYLE_EMPHASIS
    line_w = style["line_width"]
    ref_w = style["ref_line_width"]
    star_ms = float(np.sqrt(style["start_size"]) * 1.15)

    fig, ax = plt.subplots(figsize=(11.0, 0.55))
    ax.axis("off")
    handles = [
        Line2D([0], [0], color="0.15", linestyle="--", linewidth=ref_w, label="reference"),
        Line2D([0], [0], color="#2166ac", linewidth=line_w, label="diffusion plan"),
        Line2D([0], [0], color="#d95f02", linestyle="--", linewidth=line_w, label="open-loop rollout"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            markerfacecolor="black",
            markersize=star_ms,
            linestyle="None",
            label="starting point",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=PLAN_PLOT_LEGEND_FONTSIZE,
        handlelength=2.8,
        handletextpad=0.6,
        columnspacing=1.6,
    )
    fig.tight_layout(pad=0.05)
    save_plan_figure(fig, output_path, save_pdf=save_pdf)


def plot_plan_vs_rollout(
    planned_xy: np.ndarray,
    rollout_xy: np.ndarray,
    reference_xy: np.ndarray,
    init_xy: np.ndarray,
    output_path: Path,
    *,
    save_pdf: bool,
    style: dict | None = None,
) -> None:
    style = style or PLOT_STYLE_DEFAULT
    line_w = style["line_width"]
    ref_w = style["ref_line_width"]
    start_size = style["start_size"]

    x_lim, y_lim = centered_xy_limits(reference_xy, planned_xy, rollout_xy, init_xy[None, :])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(reference_xy[:, 0], reference_xy[:, 1], color="0.15", linestyle="--", linewidth=ref_w, alpha=0.85, zorder=2)
    ax.plot(planned_xy[:, 0], planned_xy[:, 1], color="#2166ac", linewidth=line_w, alpha=0.95, zorder=4)
    ax.plot(rollout_xy[:, 0], rollout_xy[:, 1], color="#d95f02", linewidth=line_w, linestyle="--", alpha=0.95, zorder=3)
    ax.scatter(
        init_xy[0],
        init_xy[1],
        color="black",
        s=start_size,
        marker="*",
        zorder=7,
        edgecolors="white",
        linewidths=0.8,
    )
    setup_plan_axes(ax, x_lim, y_lim)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.tight_layout()
    save_plan_figure(fig, output_path, save_pdf=save_pdf)


def load_candidates(traj_dir: Path) -> dict:
    data = np.load(traj_dir / "candidates.npz")
    with open(traj_dir / "candidates.meta.json") as f:
        meta = json.load(f)
    candidates = []
    n = int(data["candidate_idx"].shape[0])
    for i in range(n):
        candidates.append(
            {
                "candidate_idx": int(data["candidate_idx"][i]),
                "seed": int(data["seed"][i]),
                "plan_reward": float(data["plan_reward"][i]),
                "rollout_reward": float(data["rollout_reward"][i]),
                "feasibility_mean_l2_future": float(data["feasibility_mean_l2_future"][i]),
                "planned_xy": data["planned_xy"][i],
                "rollout_xy": data["rollout_xy"][i],
            }
        )
    return {
        "meta": meta,
        "reference_xy": data["reference_xy"],
        "init_xy": data["init_xy"],
        "candidates": candidates,
    }


def replot_run(run_dir: Path, top_k: int, *, save_pdf: bool) -> int:
    traj_dir = run_dir / "reference_plan" / "trajectories"
    plot_dir = run_dir / "reference_plan" / "plots"
    cached = load_candidates(traj_dir)
    trajectory = cached["meta"]["reference_name"]
    config = cached["meta"]["config_name"]
    style = resolve_plot_style(trajectory, config)
    ranked = sorted(cached["candidates"], key=lambda c: c["plan_reward"], reverse=True)
    top_plots = []
    for rank, cand in enumerate(ranked[: max(1, top_k)], start=1):
        plot_path = plot_dir / f"plan_{trajectory}_{config}_rank{rank:02d}_cand{cand['candidate_idx']}.png"
        plot_plan_vs_rollout(
            cand["planned_xy"],
            cand["rollout_xy"],
            cached["reference_xy"],
            cached["init_xy"],
            plot_path,
            save_pdf=save_pdf,
            style=style,
        )
        top_plots.append(
            {
                "rank": rank,
                "candidate_idx": cand["candidate_idx"],
                "seed": cand["seed"],
                "plan_reward": cand["plan_reward"],
                "rollout_reward": cand["rollout_reward"],
                "feasibility_mean_l2_future": cand["feasibility_mean_l2_future"],
                "plot_path": str(plot_path),
            }
        )

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        with open(summary_path) as f:
            payload = json.load(f)
        block = payload.get("reference_plan", {})
        block["top_candidates_by_plan_reward"] = top_plots
        block["plot_dir"] = str(plot_dir)
        block["replot_from_cache"] = True
        payload["reference_plan"] = block
        with open(summary_path, "w") as f:
            json.dump(payload, f, indent=2)
    return len(top_plots)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/unicycle_eval/reference_plan_v1"))
    parser.add_argument("--trajectories", nargs="*", default=None)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional subset of config names (e.g. standard_w_cg30).",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-pdf", action="store_true", help="Also write PDF alongside PNG.")
    parser.add_argument(
        "--legend-path",
        type=Path,
        default=None,
        help="Write standalone legend figure (default: {root}/reference_plan_legend.png).",
    )
    parser.add_argument(
        "--legend-only",
        action="store_true",
        help="Only write the standalone legend figure.",
    )
    args = parser.parse_args()

    apply_figure_plot_style()
    root = args.root.resolve()
    legend_path = args.legend_path or (root / "reference_plan_legend.png")

    if args.legend_only:
        save_plan_legend_figure(legend_path, save_pdf=args.save_pdf, style=PLOT_STYLE_EMPHASIS)
        print(f"Wrote legend {legend_path}", flush=True)
        return

    traj_filter = set(args.trajectories) if args.trajectories else None
    config_filter = set(args.configs) if args.configs else None
    npz_paths = sorted(root.glob("*/*/reference_plan/trajectories/candidates.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No candidates.npz under {root}")

    t0 = time.perf_counter()
    total = 0
    n_runs = 0
    for npz_path in npz_paths:
        run_dir = npz_path.parent.parent.parent
        trajectory = run_dir.parent.name
        config = run_dir.name
        if traj_filter and trajectory not in traj_filter:
            continue
        if config_filter and config not in config_filter:
            continue
        n = replot_run(run_dir, args.top_k, save_pdf=args.save_pdf)
        total += n
        n_runs += 1
        print(f"[replot] {trajectory}/{run_dir.name}: {n} plots", flush=True)

    save_plan_legend_figure(legend_path, save_pdf=args.save_pdf, style=PLOT_STYLE_EMPHASIS)
    print(f"Done: {total} plots across {n_runs} runs in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"Wrote legend {legend_path}", flush=True)


if __name__ == "__main__":
    main()
