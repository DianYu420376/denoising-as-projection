import json
import os
import uuid
from pathlib import Path

import d4rl
import gym
import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.diffusion.guidance import validate_guidance_config
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters
from d4rl_render_utils import (
    capture_frame,
    default_video_path,
    env_reset,
    env_step,
    is_offline_d4rl_env,
    make_sim_eval_env,
    make_video_writer,
    resolve_ckpt_stem,
    setup_headless_rendering,
)
from utils import set_episode_seed, set_seed


def _init_wandb(args):
    if not args.enable_wandb or args.mode != "train":
        return

    wandb.require("core")
    run_name = args.name
    if run_name in (None, "default", "Default"):
        run_name = f"{args.task.env_name}_seed{args.seed}"

    wandb.init(
        reinit=True,
        id=str(uuid.uuid4()),
        project=str(args.project),
        group=str(args.group),
        name=str(run_name),
        mode=str(args.wandb_mode),
        config=OmegaConf.to_container(args, resolve=True),
    )


def _resolve_save_path(args) -> str:
    base = f"results/{args.pipeline_name}/{args.task.env_name}/"
    suffix = getattr(args, "run_suffix", None)
    if suffix:
        return f"{base}{suffix}/"
    return base


def _load_checkpoints(agent, save_path: str, ckpt: str):
    ckpt_stem = resolve_ckpt_stem(str(ckpt))
    agent.load(save_path + f"diffusion_ckpt_{ckpt_stem}.pt")
    agent.classifier.load(save_path + f"classifier_ckpt_{ckpt_stem}.pt")


def _resolve_ignore_termination(args, render_enabled: bool) -> bool:
    if render_enabled:
        return bool(getattr(args, "render_ignore_termination", True))
    # Non-render inference defaults to stopping on fall for benchmark-faithful eval.
    return False


def _write_eval_output(path: str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[inference] Wrote eval results to {out_path}")


def _rollout_vector_env(env_eval, agent, normalizer, args, obs_dim, act_dim):
    episode_metrics = []
    prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
    max_steps = int(getattr(args, "max_render_steps", 1000))

    for _ in range(args.num_episodes):
        obs, ep_reward, cum_done, t = env_eval.reset(), 0.0, 0.0, 0

        while not np.all(cum_done) and t < max_steps + 1:
            obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
            prior[:, 0, :obs_dim] = obs
            traj, log = agent.sample(
                prior.repeat(args.num_candidates, 1, 1),
                solver=args.solver,
                n_samples=args.num_candidates * args.num_envs,
                sample_steps=args.sampling_steps,
                use_ema=args.use_ema,
                w_cg=args.task.w_cg,
                guidance_mode=args.guidance_mode,
                optimization_guidance_scale=args.optimization_guidance_scale,
                temperature=args.temperature,
            )

            logp = log["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
            idx = logp.argmax(0)
            act = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)[
                idx, torch.arange(args.num_envs), 0, obs_dim:
            ]
            act = act.clip(-1.0, 1.0).cpu().numpy()

            obs, rew, done, _ = env_eval.step(act)

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            ep_reward += (rew * (1 - cum_done)) if t < max_steps else rew
            print(
                f"[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}, "
                f"logp: {logp[idx, torch.arange(args.num_envs)]}"
            )

        for env_idx in range(args.num_envs):
            episode_metrics.append(
                {
                    "raw_reward": float(ep_reward[env_idx]),
                    "survival_steps": int(t),
                    "terminated_early": bool(cum_done[env_idx] and t < max_steps),
                }
            )

    return episode_metrics


def _rollout_single_env(
    env_eval,
    agent,
    normalizer,
    args,
    obs_dim,
    act_dim,
    video_writer=None,
    ignore_termination: bool | None = None,
    env_seed: int | None = None,
):
    prior = torch.zeros((1, args.task.horizon, obs_dim + act_dim), device=args.device)
    obs = env_reset(env_eval, seed=env_seed)
    ep_reward = 0.0
    total_steps = 0

    max_steps = int(getattr(args, "max_render_steps", 1000))
    reset_on_done = getattr(args, "render_reset_on_done", False)
    if ignore_termination is None:
        ignore_termination = _resolve_ignore_termination(args, video_writer is not None)

    fell = False
    while total_steps < max_steps:
        obs_norm = torch.tensor(normalizer.normalize(obs[None, :]), device=args.device, dtype=torch.float32)
        prior[:, 0, :obs_dim] = obs_norm
        traj, log = agent.sample(
            prior.repeat(args.num_candidates, 1, 1),
            solver=args.solver,
            n_samples=args.num_candidates,
            sample_steps=args.sampling_steps,
            use_ema=args.use_ema,
            w_cg=args.task.w_cg,
            guidance_mode=args.guidance_mode,
            optimization_guidance_scale=args.optimization_guidance_scale,
            temperature=args.temperature,
        )

        logp = log["log_p"].view(args.num_candidates, 1, -1).sum(-1)
        idx = logp.argmax(0)
        act = traj.view(args.num_candidates, 1, args.task.horizon, -1)[idx, 0, 0, obs_dim:]
        act = act.clip(-1.0, 1.0).cpu().numpy()

        obs, rew, done, _ = env_step(env_eval, act)
        total_steps += 1
        ep_reward += rew
        print(
            f"[step={total_steps}/{max_steps}] "
            f"rew: {np.around(rew, 2)}, logp: {float(logp[idx, 0])}, done: {done}"
        )

        if video_writer is not None:
            video_writer.append_data(capture_frame(env_eval, args.render_width, args.render_height))

        if done and not ignore_termination:
            fell = True
            if reset_on_done and total_steps < max_steps:
                obs = env_reset(env_eval, seed=env_seed)
                print(f"[render] env reset after fall (step={total_steps}/{max_steps})")
            else:
                break

    return {
        "raw_reward": float(ep_reward),
        "survival_steps": int(total_steps),
        "terminated_early": bool(fell),
    }


@hydra.main(config_path="../configs/diffuser/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):

    args.device = args.device if torch.cuda.is_available() else "cpu"
    _init_wandb(args)
    set_seed(args.seed)

    save_path = _resolve_save_path(args)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    print(f"[pipeline] Checkpoint dir: {save_path}")

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset_h5path = getattr(args, "dataset_h5path", None)
    if dataset_h5path:
        print(f"[pipeline] Loading dataset from {dataset_h5path}")
        raw_dataset = env.get_dataset(h5path=str(dataset_h5path))
    else:
        raw_dataset = env.get_dataset()
    dataset = D4RLMuJoCoDataset(
        raw_dataset,
        horizon=args.task.horizon,
        terminal_penalty=args.terminal_penalty,
        discount=args.discount,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.task.dim_mult,
        timestep_emb_type="positional",
        attention=False,
        kernel_size=5,
    )
    nn_classifier = HalfJannerUNet1d(
        args.task.horizon,
        obs_dim + act_dim,
        out_dim=1,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.task.dim_mult,
        timestep_emb_type="positional",
        kernel_size=3,
    )

    print("======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print("======================= Parameter Report of Classifier =======================")
    report_parameters(nn_classifier)
    print("==============================================================================")

    classifier = CumRewClassifier(nn_classifier, device=args.device)

    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.0
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    agent = DiscreteDiffusionSDE(
        nn_diffusion,
        None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=classifier,
        ema_rate=args.ema_rate,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
        predict_noise=args.predict_noise,
    )

    # ---------------------- Training ----------------------
    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        classifier_lr_scheduler = CosineAnnealingLR(agent.classifier.optim, args.classifier_gradient_steps)

        agent.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0.0, "avg_loss_classifier": 0.0}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device)

            x = torch.cat([obs, act], -1)

            log["avg_loss_diffusion"] += agent.update(x)["loss"]
            diffusion_lr_scheduler.step()
            if n_gradient_step <= args.classifier_gradient_steps:
                log["avg_loss_classifier"] += agent.update_classifier(x, val)["loss"]
                classifier_lr_scheduler.step()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_classifier"] /= args.log_interval
                print(log)
                if args.enable_wandb:
                    wandb.log(log, step=n_gradient_step + 1)
                log = {"avg_loss_diffusion": 0.0, "avg_loss_classifier": 0.0}

            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.classifier.save(save_path + f"classifier_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")
                agent.classifier.save(save_path + f"classifier_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

        if n_gradient_step > 0:
            agent.save(save_path + "diffusion_ckpt_latest.pt")
            agent.classifier.save(save_path + "classifier_ckpt_latest.pt")
            print(f"Saved final checkpoints to {save_path}")

        if args.enable_wandb:
            wandb.finish()

    # ---------------------- Inference / Rendering ----------------------
    elif args.mode in ("inference", "render"):

        render_enabled = args.mode == "render" or bool(args.render_video)
        if render_enabled:
            setup_headless_rendering()

        _load_checkpoints(agent, save_path, args.ckpt)
        agent.eval()
        normalizer = dataset.get_normalizer()

        validate_guidance_config(
            str(args.guidance_mode),
            float(args.task.w_cg),
            float(args.optimization_guidance_scale),
        )
        print(
            "[inference] Guidance config: "
            f"guidance_mode={args.guidance_mode}, "
            f"optimization_guidance_scale={args.optimization_guidance_scale}, "
            f"w_cg={args.task.w_cg}"
        )

        dataset_env = gym.make(args.task.env_name)
        use_sim_fallback = is_offline_d4rl_env(dataset_env)
        score_env = dataset_env
        dataset_env.close()

        if render_enabled or use_sim_fallback:
            if args.num_envs != 1 and render_enabled:
                print("[render] Forcing num_envs=1 for video recording.")
            num_envs = 1 if render_enabled else args.num_envs
            if use_sim_fallback and num_envs != 1:
                print("[render] Offline D4RL task detected; using single-env sim fallback.")
                num_envs = 1

            ignore_termination = _resolve_ignore_termination(args, render_enabled)
            env_eval, sim_name = make_sim_eval_env(
                args.task.env_name,
                sim_env_name=args.sim_env_name,
                render=render_enabled,
                render_width=args.render_width,
                render_height=args.render_height,
                ignore_termination=ignore_termination,
            )
            print(
                f"[inference] Rollout sim env: {sim_name} "
                f"(ignore_termination={ignore_termination})"
            )

            writers = []
            if render_enabled:
                for episode_idx in range(args.num_episodes):
                    video_path = default_video_path(
                        args.video_dir, args.pipeline_name, args.task.env_name, episode_idx
                    )
                    print(f"[render] Saving video to {video_path}")
                    writers.append(make_video_writer(video_path, args.render_fps))

            episode_metrics = []
            for episode_idx in range(args.num_episodes):
                episode_seed = int(args.seed) + episode_idx
                set_episode_seed(episode_seed)
                writer = writers[episode_idx] if writers else None
                ep_metrics = _rollout_single_env(
                    env_eval,
                    agent,
                    normalizer,
                    args,
                    obs_dim,
                    act_dim,
                    video_writer=writer,
                    ignore_termination=ignore_termination,
                    env_seed=episode_seed,
                )
                episode_metrics.append(ep_metrics)
                if writer is not None:
                    writer.close()
                    print(f"[render] Saved episode {episode_idx} video.")

            env_eval.close()
        else:
            env_eval = gym.vector.make(args.task.env_name, args.num_envs)
            episode_metrics = _rollout_vector_env(env_eval, agent, normalizer, args, obs_dim, act_dim)
            env_eval.close()

        episodes = []
        for ep_idx, metrics in enumerate(episode_metrics):
            normalized = float(score_env.get_normalized_score(metrics["raw_reward"]))
            episodes.append(
                {
                    "episode": ep_idx,
                    "raw_reward": metrics["raw_reward"],
                    "normalized_score": normalized,
                    "normalized_score_x100": normalized * 100.0,
                    "survival_steps": metrics["survival_steps"],
                    "terminated_early": metrics["terminated_early"],
                }
            )

        normalized_scores = np.array([ep["normalized_score"] for ep in episodes], dtype=np.float64)
        raw_rewards = np.array([ep["raw_reward"] for ep in episodes], dtype=np.float64)
        survival_steps = np.array([ep["survival_steps"] for ep in episodes], dtype=np.float64)

        summary = {
            "task": args.task.env_name,
            "ckpt": str(args.ckpt),
            "seed": int(args.seed),
            "guidance_mode": str(args.guidance_mode),
            "optimization_guidance_scale": float(args.optimization_guidance_scale),
            "w_cg": float(args.task.w_cg),
            "num_episodes": int(args.num_episodes),
            "num_candidates": int(args.num_candidates),
            "max_steps": int(getattr(args, "max_render_steps", 1000)),
            "mean_normalized_score": float(normalized_scores.mean()),
            "std_normalized_score": float(normalized_scores.std()),
            "mean_normalized_score_x100": float(normalized_scores.mean() * 100.0),
            "std_normalized_score_x100": float(normalized_scores.std() * 100.0),
            "mean_raw_reward": float(raw_rewards.mean()),
            "std_raw_reward": float(raw_rewards.std()),
            "mean_survival_steps": float(survival_steps.mean()),
            "std_survival_steps": float(survival_steps.std()),
            "episodes": episodes,
        }

        print(
            "[inference] mean_normalized_score="
            f"{summary['mean_normalized_score']:.4f} "
            f"std={summary['std_normalized_score']:.4f} "
            f"(x100: {summary['mean_normalized_score_x100']:.2f} "
            f"± {summary['std_normalized_score_x100']:.2f})"
        )
        print(
            "[inference] mean_raw_reward="
            f"{summary['mean_raw_reward']:.2f} "
            f"mean_survival_steps={summary['mean_survival_steps']:.1f}"
        )

        eval_output = getattr(args, "eval_output", None)
        if eval_output:
            _write_eval_output(str(eval_output), summary)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
