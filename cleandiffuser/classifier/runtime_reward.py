"""Analytic reward classifiers for runtime guidance (no training)."""

from __future__ import annotations

from typing import Callable, Optional

import torch


class RuntimeRewardClassifier:
    """Classifier-guidance plug-in backed by a user-defined reward function.

    ``reward_fn(x, c)`` should return a per-batch reward tensor of shape ``(batch,)``.
    ``x`` is the full trajectory tensor ``(batch, horizon, dim)`` in normalized space.

    Unlike trained log-probability classifiers, this exposes the raw reward gradient
    ``∇_x R(x)`` (not ``∇_x log R(x)``) for guidance.
    """

    uses_reward_gradient = True

    def __init__(
        self,
        reward_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        device: str = "cpu",
    ):
        self.reward_fn = reward_fn
        self.device = device

    def eval(self):
        return self

    def train(self):
        return self

    @property
    def model(self):
        """Compatibility with DiffusionModel.train/eval calling classifier.model."""
        return self

    def reward(self, x: torch.Tensor, noise: torch.Tensor, c=None):
        reward = self.reward_fn(x, c)
        if reward.ndim == 0:
            reward = reward.unsqueeze(0)
        return reward.reshape(-1, 1)

    def logp(self, x: torch.Tensor, noise: torch.Tensor, c=None):
        """Backward-compatible alias; returns reward (not log reward)."""
        return self.reward(x, noise, c)

    def reward_gradients(self, x: torch.Tensor, noise: torch.Tensor, c=None):
        x = x.clone().detach().requires_grad_(True)
        reward = self.reward(x, noise, c)
        grad = torch.autograd.grad([reward.sum()], [x], retain_graph=False)[0]
        return reward.detach(), grad.detach()

    def gradients(self, x: torch.Tensor, noise: torch.Tensor, c=None):
        return self.reward_gradients(x, noise, c)

    def save(self, path: str):
        raise NotImplementedError("RuntimeRewardClassifier has no trainable weights.")

    def load(self, path: str):
        raise NotImplementedError("RuntimeRewardClassifier has no trainable weights.")
