import json
import os
import uuid
from pathlib import Path

import gym
import h5py
import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import cleandiffuser.env.unicycle  # noqa: F401
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.dataset.unicycle_dataset import UnicycleDataset, load_unicycle_hdf5
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters
from d4rl_render_utils import resolve_ckpt_stem
from utils import set_seed


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


def _default_dataset_path() -> str:
    return "results/unicycle_offline/unicycle_offline.hdf5"


def _load_diffusion_checkpoint(agent, save_path: str, ckpt: str):
    ckpt_stem = resolve_ckpt_stem(str(ckpt))
    agent.load(save_path + f"diffusion_ckpt_{ckpt_stem}.pt")


def _write_eval_output(path: str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[inference] Wrote eval results to {out_path}")


def _clip_actions(actions: np.ndarray, env: gym.Env) -> np.ndarray:
    unwrapped = env.unwrapped
    v_bounds = unwrapped.v_bounds
    w_bounds = unwrapped.w_bounds
    clipped = np.asarray(actions, dtype=np.float32).copy()
    clipped[..., 0] = np.clip(clipped[..., 0], v_bounds[0], v_bounds[1])
    clipped[..., 1] = np.clip(clipped[..., 1], w_bounds[0], w_bounds[1])
    return clipped


def _dataset_require_full_horizon(h5path: str) -> bool:
    try:
        with h5py.File(h5path, "r") as f:
            meta = json.loads(f.attrs.get("metadata_json", "{}"))
        return bool(meta.get("require_full_horizon", False))
    except Exception:
        return False


@hydra.main(config_path="../configs/diffuser/unicycle", config_name="unicycle", version_base=None)
def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    _init_wandb(args)
    set_seed(args.seed)

    save_path = _resolve_save_path(args)
    os.makedirs(save_path, exist_ok=True)
    print(f"[pipeline] Checkpoint dir: {save_path}")

    dataset_h5path = getattr(args, "dataset_h5path", None) or _default_dataset_path()
    print(f"[pipeline] Loading dataset from {dataset_h5path}")
    raw_dataset = load_unicycle_hdf5(dataset_h5path)
    require_full = _dataset_require_full_horizon(dataset_h5path)
    dataset = UnicycleDataset(
        raw_dataset,
        horizon=args.task.horizon,
        terminal_penalty=args.terminal_penalty,
        discount=args.discount,
        require_full_horizon=require_full,
    )
    if require_full:
        print(f"[pipeline] require_full_horizon=True, training sequences={len(dataset)}")
    elif dataset.skipped_short_episodes:
        print(f"[pipeline] skipped {dataset.skipped_short_episodes} short episodes")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim,
        model_dim=args.model_dim,
        emb_dim=args.model_dim,
        dim_mult=args.task.dim_mult,
        timestep_emb_type="positional",
        attention=False,
        kernel_size=5,
    )

    print("======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print("==============================================================================")

    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.0
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    agent = DiscreteDiffusionSDE(
        nn_diffusion,
        None,
        fix_mask=fix_mask,
        loss_weight=loss_weight,
        classifier=None,
        ema_rate=args.ema_rate,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
        predict_noise=args.predict_noise,
    )

    if args.mode == "train":
        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        agent.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0.0}

        for batch in loop_dataloader(dataloader):
            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            x = torch.cat([obs, act], -1)

            log["avg_loss_diffusion"] += agent.update(x)["loss"]
            diffusion_lr_scheduler.step()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                print(log)
                if args.enable_wandb:
                    wandb.log(log, step=n_gradient_step + 1)
                log = {"avg_loss_diffusion": 0.0}

            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + "diffusion_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

        if n_gradient_step > 0:
            agent.save(save_path + "diffusion_ckpt_latest.pt")
            print(f"Saved final checkpoints to {save_path}")

        if args.enable_wandb:
            wandb.finish()

    elif args.mode == "inference":
        _load_diffusion_checkpoint(agent, save_path, args.ckpt)
        agent.eval()
        normalizer = dataset.get_normalizer()

        env = gym.make(args.task.env_name)
        prior = torch.zeros((1, args.task.horizon, obs_dim + act_dim), device=args.device)
        episode_metrics = []

        for episode_idx in range(args.num_episodes):
            obs = env.reset()
            ep_reward = 0.0
            total_steps = 0
            max_steps = args.task.horizon

            while total_steps < max_steps:
                obs_norm = torch.tensor(normalizer.normalize(obs[None, :]), device=args.device, dtype=torch.float32)
                prior.zero_()
                prior[:, 0, :obs_dim] = obs_norm

                traj, _ = agent.sample(
                    prior,
                    solver=args.solver,
                    n_samples=args.num_candidates,
                    sample_steps=args.sampling_steps,
                    use_ema=args.use_ema,
                    w_cg=args.task.w_cg,
                    guidance_mode=args.guidance_mode,
                    optimization_guidance_scale=args.optimization_guidance_scale,
                    optimization_guidance_last_steps=args.optimization_guidance_last_steps,
                    temperature=args.temperature,
                )

                plan = traj[0].detach().cpu().numpy()
                act = _clip_actions(plan[0, obs_dim:], env)
                obs, rew, done, _ = env.step(act)
                total_steps += 1
                ep_reward += rew
                if done:
                    break

            episode_metrics.append(
                {
                    "episode": episode_idx,
                    "raw_reward": float(ep_reward),
                    "survival_steps": int(total_steps),
                }
            )

        env.close()
        summary = {
            "task": args.task.env_name,
            "ckpt": str(args.ckpt),
            "seed": int(args.seed),
            "episodes": episode_metrics,
        }
        print(summary)
        eval_output = getattr(args, "eval_output", None)
        if eval_output:
            _write_eval_output(str(eval_output), summary)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
