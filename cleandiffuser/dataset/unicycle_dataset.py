from typing import Dict

import numpy as np
import torch

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.utils import GaussianNormalizer, dict_apply


def load_unicycle_hdf5(path: str) -> Dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        return {key: np.asarray(f[key][:]) for key in f.keys()}


class UnicycleDataset(BaseDataset):
    """Offline unicycle trajectories chunked into fixed-length sequences."""

    def __init__(
        self,
        dataset: Dict[str, np.ndarray],
        horizon: int = 64,
        terminal_penalty: float | None = None,
        discount: float = 0.99,
        require_full_horizon: bool = False,
    ):
        super().__init__()

        observations = dataset["observations"].astype(np.float32)
        actions = dataset["actions"].astype(np.float32)
        rewards = dataset["rewards"].astype(np.float32)
        terminals = dataset["terminals"].astype(bool)
        timeouts = dataset["timeouts"].astype(bool)

        if "episode_ends" in dataset:
            episode_ends = dataset["episode_ends"].astype(np.int64)
            n_paths = len(episode_ends)
            max_path_length = int(np.max(np.diff(np.concatenate([[0], episode_ends]))))
        else:
            episode_ends = np.where(terminals | timeouts)[0] + 1
            n_paths = len(episode_ends)
            max_path_length = int(np.max(np.diff(np.concatenate([[0], episode_ends]))))

        self.normalizers = {"state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]
        self.require_full_horizon = require_full_horizon
        self.skipped_short_episodes = 0

        valid_paths: list[tuple[int, int, int]] = []
        ptr = 0
        for path_idx, ep_end in enumerate(episode_ends):
            ep_end = int(ep_end)
            path_length = ep_end - ptr
            if path_length <= 0:
                ptr = ep_end
                continue

            if require_full_horizon and path_length < horizon:
                self.skipped_short_episodes += 1
                ptr = ep_end
                continue

            valid_paths.append((path_idx, ptr, ep_end, path_length))
            ptr = ep_end

        n_valid = len(valid_paths)
        if n_valid == 0:
            raise ValueError("No episodes satisfy the dataset horizon requirements.")

        max_path_length = max(p[3] for p in valid_paths)
        self.seq_obs = np.zeros((n_valid, max_path_length, self.o_dim), dtype=np.float32)
        self.seq_act = np.zeros((n_valid, max_path_length, self.a_dim), dtype=np.float32)
        self.seq_rew = np.zeros((n_valid, max_path_length, 1), dtype=np.float32)
        self.seq_val = np.zeros((n_valid, max_path_length, 1), dtype=np.float32)
        self.indices: list[tuple[int, int, int]] = []

        for seq_idx, (path_idx, ptr_start, ep_end, path_length) in enumerate(valid_paths):
            if terminals[ep_end - 1] and not timeouts[ep_end - 1] and terminal_penalty is not None:
                rewards[ep_end - 1] = terminal_penalty

            self.seq_obs[seq_idx, :path_length] = normed_observations[ptr_start:ep_end]
            self.seq_act[seq_idx, :path_length] = actions[ptr_start:ep_end]
            self.seq_rew[seq_idx, :path_length] = rewards[ptr_start:ep_end][:, None]

            max_start = min(path_length - 1, max_path_length - horizon)
            self.indices += [(seq_idx, start, start + horizon) for start in range(max_start + 1)]

        self.seq_val[:, -1] = self.seq_rew[:, -1]
        for i in range(max_path_length - 1):
            self.seq_val[:, -2 - i] = self.seq_rew[:, -2 - i] + discount * self.seq_val[:, -1 - i]

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]
        data = {
            "obs": {"state": self.seq_obs[path_idx, start:end]},
            "act": self.seq_act[path_idx, start:end],
            "rew": self.seq_rew[path_idx, start:end],
            "val": self.seq_val[path_idx, start],
        }
        return dict_apply(data, torch.tensor)
