"""Lagrangian PPO for budget-constrained RMAB policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.signal
import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR

from ..env import RMABEnv
from .networks import LambdaNet, PolicyNet, ValueNet


@dataclass
class TrainerConfig:
    """Hyperparameters for :class:`Trainer`. Defaults target small synthetic problems; tune for larger ones."""

    epochs: int = 50
    steps_per_epoch: int = 200
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    v_lr: float = 1e-3
    lambda_lr: float = 5e-2
    train_pi_iters: int = 20
    train_v_iters: int = 20
    target_kl: float = 0.05
    entropy_coef: float = 0.0
    lambda_update_every: int = 4
    lambda_lr_decay: float = 0.995
    final_value_only_epochs: int = 5
    hidden: tuple[int, ...] = (64, 64)
    lambda_hidden: tuple[int, ...] = (8, 8)
    seed: int = 0


def _discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """Discounted cumulative sum: ``out[t] = sum_{k>=0} discount^k * x[t+k]``."""
    return scipy.signal.lfilter([1.0], [1.0, -discount], x[::-1])[::-1]


class Trainer:
    """PPO with a learned Lagrange multiplier for the RMAB budget constraint."""

    def __init__(self, env: RMABEnv, config: TrainerConfig | None = None) -> None:
        self.env = env
        self.cfg = config or TrainerConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        ds = env.dataset
        self.N, self.S, self.A = ds.n_arms, ds.n_states, ds.n_actions
        self.tp_feat_dim = self.S * self.A * self.S

        self.policy = PolicyNet(self.S, self.A, self.tp_feat_dim, self.cfg.hidden)
        self.value = ValueNet(self.S, self.tp_feat_dim, self.cfg.hidden)
        self.lambda_net = LambdaNet(self.N, self.tp_feat_dim, self.cfg.lambda_hidden)

        self.pi_opt = Adam(self.policy.parameters(), lr=self.cfg.pi_lr)
        self.v_opt = Adam(self.value.parameters(), lr=self.cfg.v_lr)
        self.l_opt = SGD(self.lambda_net.parameters(), lr=self.cfg.lambda_lr)
        self.l_sched = ExponentialLR(self.l_opt, gamma=self.cfg.lambda_lr_decay)

        self._tp_feats_per_arm = torch.as_tensor(
            ds.transition_matrices.reshape(self.N, -1), dtype=torch.float32
        )
        self._action_costs_t = torch.as_tensor(ds.action_costs, dtype=torch.float32)

    def _policy_obs(self, state: np.ndarray, lamb: float) -> torch.Tensor:
        idx = torch.as_tensor(state, dtype=torch.long)
        ohs = torch.zeros((self.N, self.S), dtype=torch.float32)
        ohs[torch.arange(self.N), idx] = 1.0
        lam = torch.full((self.N, 1), float(lamb), dtype=torch.float32)
        return torch.cat([ohs, lam, self._tp_feats_per_arm], dim=1)

    def _lambda_input(self, state: np.ndarray) -> torch.Tensor:
        s = torch.as_tensor(state, dtype=torch.float32)
        return torch.cat([s, self._tp_feats_per_arm.flatten()])

    def _current_lambda(self, state: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.lambda_net(self._lambda_input(state)))

    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select actions for one timestep.

        With ``deterministic=True``, returns the budget-respecting greedy
        assignment. With ``deterministic=False``, samples per-arm from the
        policy distribution (the budget is *not* enforced; this is what
        the trainer uses during rollouts).
        """
        lamb = self._current_lambda(state)
        with torch.no_grad():
            dist = self.policy(self._policy_obs(state, lamb))
            if not deterministic:
                return dist.sample().cpu().numpy().astype(np.int64)
            probs = dist.probs.cpu().numpy()
        return self._greedy_budget_actions(probs)

    def _greedy_budget_actions(self, probs: np.ndarray) -> np.ndarray:
        costs = self.env.dataset.action_costs
        budget = self.env.budget
        cheapest = int(np.argmin(costs))
        actions = np.full(self.N, cheapest, dtype=np.int64)
        spent = float(costs[cheapest]) * self.N
        if spent > budget + 1e-9:
            raise ValueError("Cheapest action per arm already exceeds budget")
        while True:
            best_gain = 1e-12
            best = None
            cur_costs = costs[actions]
            for i in range(self.N):
                for a in range(self.A):
                    if a == actions[i]:
                        continue
                    dc = float(costs[a] - cur_costs[i])
                    if spent + dc > budget + 1e-9:
                        continue
                    gain = float(probs[i, a] - probs[i, actions[i]])
                    if gain > best_gain:
                        best_gain, best = gain, (i, a, dc)
            if best is None:
                break
            i, a, dc = best
            actions[i] = a
            spent += dc
        return actions

    def _collect_rollout(
        self, init_state: np.ndarray, lamb: float
    ) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        cfg = self.cfg
        T, N, S = cfg.steps_per_epoch, self.N, self.S
        obs_dim = S + 1 + self.tp_feat_dim
        obs = torch.zeros((T, N, obs_dim), dtype=torch.float32)
        act = torch.zeros((T, N), dtype=torch.long)
        logp = torch.zeros((T, N), dtype=torch.float32)
        val = torch.zeros((T, N), dtype=torch.float32)
        rew = torch.zeros((T, N), dtype=torch.float32)
        cost = torch.zeros((T, N), dtype=torch.float32)

        state = init_state
        for t in range(T):
            with torch.no_grad():
                o = self._policy_obs(state, lamb)
                dist = self.policy(o)
                a = dist.sample()
                lp = dist.log_prob(a)
                v = self.value(o)
            obs[t] = o
            act[t] = a
            logp[t] = lp
            val[t] = v
            cost[t] = self._action_costs_t[a]
            a_np = a.cpu().numpy().astype(np.int64)
            next_state, r_np, _, _ = self.env.step(a_np, enforce_budget=False)
            rew[t] = torch.as_tensor(r_np, dtype=torch.float32)
            state = next_state

        with torch.no_grad():
            last_val = self.value(self._policy_obs(state, lamb))

        rollout = {
            "obs": obs,
            "act": act,
            "logp": logp,
            "val": val,
            "rew": rew,
            "cost": cost,
            "last_val": last_val,
        }
        return rollout, state

    def _compute_advantages(self, rollout: dict[str, torch.Tensor], lamb: float) -> None:
        cfg = self.cfg
        adj_rew = rollout["rew"] - lamb * rollout["cost"]
        val = rollout["val"]
        last_val = rollout["last_val"]
        T, N = adj_rew.shape

        adv = np.zeros((T, N), dtype=np.float32)
        ret = np.zeros((T, N), dtype=np.float32)
        for i in range(N):
            r = adj_rew[:, i].numpy()
            v = val[:, i].numpy()
            lv = float(last_val[i])
            v_aug = np.concatenate([v, [lv]])
            deltas = r + cfg.gamma * v_aug[1:] - v_aug[:-1]
            adv[:, i] = _discount_cumsum(deltas, cfg.gamma * cfg.gae_lambda)
            ret[:, i] = _discount_cumsum(np.concatenate([r, [lv]]), cfg.gamma)[:-1]

        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        adv_mean = adv_t.mean(dim=0, keepdim=True)
        adv_std = adv_t.std(dim=0, keepdim=True)
        adv_t = (adv_t - adv_mean) / (adv_std + 1e-8)
        rollout["adv"] = adv_t
        rollout["ret"] = torch.as_tensor(ret, dtype=torch.float32)

    def _update_policy(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        obs = rollout["obs"].reshape(-1, rollout["obs"].shape[-1])
        act = rollout["act"].reshape(-1)
        adv = rollout["adv"].reshape(-1)
        old_logp = rollout["logp"].reshape(-1)
        last_loss, last_kl = 0.0, 0.0
        for _ in range(cfg.train_pi_iters):
            self.pi_opt.zero_grad()
            dist = self.policy(obs)
            new_logp = dist.log_prob(act)
            ratio = torch.exp(new_logp - old_logp)
            clipped = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv
            pg_loss = -torch.min(ratio * adv, clipped).mean()
            loss = pg_loss - cfg.entropy_coef * dist.entropy().mean()
            loss.backward()
            self.pi_opt.step()
            with torch.no_grad():
                last_kl = float((old_logp - new_logp).mean())
                last_loss = float(pg_loss.detach())
            if last_kl > 1.5 * cfg.target_kl:
                break
        return {"pi_loss": last_loss, "kl": last_kl}

    def _update_value(self, rollout: dict[str, torch.Tensor]) -> float:
        obs = rollout["obs"].reshape(-1, rollout["obs"].shape[-1])
        ret = rollout["ret"].reshape(-1)
        last = 0.0
        for _ in range(self.cfg.train_v_iters):
            self.v_opt.zero_grad()
            loss = ((self.value(obs) - ret) ** 2).mean()
            loss.backward()
            self.v_opt.step()
            last = float(loss.detach())
        return last

    def _update_lambda(
        self, rollout: dict[str, torch.Tensor], init_state: np.ndarray
    ) -> dict[str, float]:
        arm_summed = rollout["cost"].sum(dim=1).numpy()
        disc_cost = float(_discount_cumsum(arm_summed, self.cfg.gamma)[0])
        target = self.env.budget / (1.0 - self.cfg.gamma)
        self.l_opt.zero_grad()
        lam = self.lambda_net(self._lambda_input(init_state))
        loss = lam * (target - disc_cost)
        loss.backward()
        self.l_opt.step()
        self.l_sched.step()
        return {"lambda_loss": float(loss.detach()), "disc_cost": disc_cost}

    def train(
        self,
        callback: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Run the training loop. Returns a per-epoch history."""
        cfg = self.cfg
        history: list[dict[str, Any]] = []
        for epoch in range(cfg.epochs):
            state = self.env.reset(randomize=True)
            lamb = self._current_lambda(state)
            init_state = state.copy()
            rollout, state = self._collect_rollout(state, lamb)
            self._compute_advantages(rollout, lamb)

            pi_info = self._update_policy(rollout)
            v_loss = self._update_value(rollout)

            lam_info: dict[str, float] = {}
            in_value_only = (cfg.epochs - epoch) <= cfg.final_value_only_epochs
            if epoch > 0 and epoch % cfg.lambda_update_every == 0 and not in_value_only:
                lam_info = self._update_lambda(rollout, init_state)

            info: dict[str, Any] = {
                "epoch": epoch,
                "lambda": lamb,
                "mean_reward": float(rollout["rew"].mean()),
                "mean_cost_per_step": float(rollout["cost"].sum(dim=1).mean()),
                "value_loss": v_loss,
                **pi_info,
                **lam_info,
            }
            history.append(info)
            if callback is not None:
                callback(epoch, info)
        return history
