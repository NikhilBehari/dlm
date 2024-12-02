"""Policy, value, and Lagrange-multiplier networks for RMAB-PPO."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def _mlp(sizes: list[int], activation=nn.Tanh, output_activation=nn.Identity) -> nn.Sequential:
    layers: list[nn.Module] = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    """Per-arm policy with shared weights."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        tp_feat_dim: int,
        hidden: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.net = _mlp([n_states + 1 + tp_feat_dim, *hidden, n_actions])

    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(x))


class ValueNet(nn.Module):
    """Per-arm critic with shared weights."""

    def __init__(
        self,
        n_states: int,
        tp_feat_dim: int,
        hidden: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.net = _mlp([n_states + 1 + tp_feat_dim, *hidden, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LambdaNet(nn.Module):
    """Lagrange multiplier conditioned on full system state and dynamics.

    The output bias is initialized so ``softplus(output) ≈ init_lambda`` at
    the start of training, keeping the cost penalty small enough for the
    policy to explore pull actions before the multiplier ramps up.
    """

    def __init__(
        self,
        n_arms: int,
        tp_feat_dim: int,
        hidden: tuple[int, ...] = (8, 8),
        init_lambda: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = _mlp([n_arms + n_arms * tp_feat_dim, *hidden, 1])
        last_linear = next(m for m in reversed(self.net) if isinstance(m, nn.Linear))
        with torch.no_grad():
            last_linear.weight.zero_()
            last_linear.bias.fill_(math.log(math.expm1(init_lambda)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x).squeeze(-1))
