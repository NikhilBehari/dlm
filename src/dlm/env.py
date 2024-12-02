"""Restless multi-armed bandit dataset, environment, and constructors."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# RewardFn signature: (arm_features: np.ndarray, next_state: int) -> reward
RewardFn = Callable[[np.ndarray, int], float]


@dataclass(frozen=True)
class RMABDataset:
    """Static description of an RMAB problem."""

    transition_matrices: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...] = ()
    feature_descriptions: tuple[str, ...] = ()
    feature_groups: tuple[tuple[str, tuple[int, ...]], ...] = ()
    action_costs: np.ndarray | None = None

    def __post_init__(self) -> None:
        T = np.asarray(self.transition_matrices, dtype=np.float64)
        F = np.asarray(self.features, dtype=np.float32)
        if T.ndim != 4:
            raise ValueError(f"transition_matrices must be (N,S,A,S); got shape {T.shape}")
        if F.ndim != 2:
            raise ValueError(f"features must be (N,F); got shape {F.shape}")
        if T.shape[0] != F.shape[0]:
            raise ValueError(f"arm-count mismatch: T has {T.shape[0]}, features has {F.shape[0]}")
        if T.shape[1] != T.shape[3]:
            raise ValueError(f"transition tensor must be square in state dims; got {T.shape}")
        if not np.allclose(T.sum(axis=-1), 1.0, atol=1e-6):
            raise ValueError("transition matrices must sum to 1 along the last axis")
        object.__setattr__(self, "transition_matrices", T)
        object.__setattr__(self, "features", F)
        if self.action_costs is None:
            object.__setattr__(self, "action_costs", np.arange(T.shape[2], dtype=np.float64))
        else:
            ac = np.asarray(self.action_costs, dtype=np.float64)
            if ac.shape != (T.shape[2],):
                raise ValueError(f"action_costs must be ({T.shape[2]},); got {ac.shape}")
            object.__setattr__(self, "action_costs", ac)
        if self.feature_names and len(self.feature_names) != F.shape[1]:
            raise ValueError(
                f"feature_names length {len(self.feature_names)} != n_features {F.shape[1]}"
            )
        if self.feature_descriptions and len(self.feature_descriptions) != F.shape[1]:
            raise ValueError(
                f"feature_descriptions length {len(self.feature_descriptions)} != "
                f"n_features {F.shape[1]}"
            )
        if self.feature_groups:
            seen_names: set[str] = set()
            for name, idxs in self.feature_groups:
                if name in seen_names:
                    raise ValueError(f"duplicate feature_groups entry: {name!r}")
                seen_names.add(name)
                for j in idxs:
                    if not 0 <= j < F.shape[1]:
                        raise ValueError(
                            f"feature_groups[{name!r}] contains out-of-range index {j}"
                        )

    @property
    def n_arms(self) -> int:
        return self.transition_matrices.shape[0]

    @property
    def n_states(self) -> int:
        return self.transition_matrices.shape[1]

    @property
    def n_actions(self) -> int:
        return self.transition_matrices.shape[2]

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

    @classmethod
    def load(cls, env_dir: str | Path) -> RMABDataset:
        """Load a dataset from a standardized env directory.

        Layout (any unspecified field falls back to its default)::

            env_dir/
            ├── env.json          {"feature_names": [...], "feature_descriptions": [...],
            │                      "action_costs": [...]}
            ├── transitions.npy   (N, S, A, S) array
            └── features.npy      (N, F) array
        """
        path = Path(env_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"env directory not found: {path}")
        transitions = np.load(path / "transitions.npy")
        features = np.load(path / "features.npy")
        manifest_path = path / "env.json"
        manifest: dict[str, Any] = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )
        return cls(
            transition_matrices=transitions,
            features=features,
            feature_names=tuple(manifest.get("feature_names", ())),
            feature_descriptions=tuple(manifest.get("feature_descriptions", ())),
            action_costs=manifest.get("action_costs"),
        )

    def save(self, env_dir: str | Path) -> None:
        """Persist the dataset to a directory in the standard layout."""
        path = Path(env_dir)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "transitions.npy", self.transition_matrices)
        np.save(path / "features.npy", self.features)
        manifest: dict[str, Any] = {"action_costs": self.action_costs.tolist()}
        if self.feature_names:
            manifest["feature_names"] = list(self.feature_names)
        if self.feature_descriptions:
            manifest["feature_descriptions"] = list(self.feature_descriptions)
        (path / "env.json").write_text(json.dumps(manifest, indent=2))


class RMABEnv:
    """Stateful simulator wrapping an :class:`RMABDataset` with a reward and budget."""

    def __init__(
        self,
        dataset: RMABDataset,
        budget: float,
        reward_fn: RewardFn,
        seed: int | None = None,
    ) -> None:
        if budget <= 0:
            raise ValueError(f"budget must be positive; got {budget}")
        self.dataset = dataset
        self.budget = float(budget)
        self.reward_fn = reward_fn
        self.rng = np.random.default_rng(seed)
        self._state = np.zeros(dataset.n_arms, dtype=np.int64)

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def reset(self, seed: int | None = None, randomize: bool = False) -> np.ndarray:
        """Reset arm states. Returns the new state vector."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if randomize:
            self._state = self.rng.integers(0, self.dataset.n_states, size=self.dataset.n_arms)
        else:
            self._state = np.zeros(self.dataset.n_arms, dtype=np.int64)
        return self.state

    def step(
        self,
        action: np.ndarray,
        enforce_budget: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        """Advance one step. Returns ``(next_state, rewards, done, info)``.

        With ``enforce_budget=True`` (the default), actions whose total cost
        exceeds the budget raise :class:`ValueError`. The PPO trainer passes
        ``False`` during rollouts and relies on the Lagrangian penalty
        instead.
        """
        ds = self.dataset
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (ds.n_arms,):
            raise ValueError(f"action must have shape ({ds.n_arms},); got {action.shape}")
        if action.min() < 0 or action.max() >= ds.n_actions:
            raise ValueError(f"action values must lie in [0,{ds.n_actions})")

        cost = float(ds.action_costs[action].sum())
        if enforce_budget and cost > self.budget + 1e-9:
            raise ValueError(f"action cost {cost} exceeds budget {self.budget}")

        next_state = np.empty(ds.n_arms, dtype=np.int64)
        for i in range(ds.n_arms):
            p = ds.transition_matrices[i, self._state[i], action[i]]
            next_state[i] = self.rng.choice(ds.n_states, p=p)

        rewards = np.fromiter(
            (self.reward_fn(ds.features[i], int(next_state[i])) for i in range(ds.n_arms)),
            dtype=np.float32,
            count=ds.n_arms,
        )
        self._state = next_state
        return self.state, rewards, False, {"cost": cost}


@dataclass
class Arm:
    """One arm of an RMAB.

    Args:
        features: 1-D feature vector consumed by the reward function.
        transitions: ``(S, A, S)`` array where ``transitions[s, a, s']``
            is ``P(next_state = s' | state = s, action = a)``.
            Rows along the last axis must sum to 1.
    """

    features: Sequence[float] | np.ndarray
    transitions: np.ndarray


def from_arms(
    arms: Sequence[Arm],
    feature_names: Sequence[str] = (),
    feature_descriptions: Sequence[str] = (),
    action_costs: Sequence[float] | None = None,
) -> RMABDataset:
    """Build an :class:`RMABDataset` from a list of per-arm :class:`Arm` specs.

    All arms must share the same ``(S, A)`` shape and the same feature
    dimensionality. Validation is performed by :class:`RMABDataset`.
    """
    if not arms:
        raise ValueError("at least one arm required")
    features = np.array([list(a.features) for a in arms], dtype=np.float32)
    transitions = np.stack([np.asarray(a.transitions, dtype=np.float64) for a in arms], axis=0)
    return RMABDataset(
        transition_matrices=transitions,
        features=features,
        feature_names=tuple(feature_names),
        feature_descriptions=tuple(feature_descriptions),
        action_costs=np.asarray(action_costs, dtype=np.float64) if action_costs is not None else None,
    )


def binary_arm(
    features: Sequence[float] | np.ndarray,
    p_pull_at_0: float,
    p_pull_at_1: float = 0.9,
    p_pass_at_0: float = 0.3,
    p_pass_at_1: float = 0.5,
) -> Arm:
    """Convenience constructor for a 2-state, 2-action arm.

    Each ``p_*`` is the probability of transitioning to state 1 from the
    given ``(state, action)`` pair. The complementary probability of
    transitioning to state 0 is computed automatically.
    """
    T = np.zeros((2, 2, 2), dtype=np.float64)
    T[0, 0] = [1 - p_pass_at_0, p_pass_at_0]
    T[1, 0] = [1 - p_pass_at_1, p_pass_at_1]
    T[0, 1] = [1 - p_pull_at_0, p_pull_at_0]
    T[1, 1] = [1 - p_pull_at_1, p_pull_at_1]
    return Arm(features=features, transitions=T)


def synthetic_dataset(
    n_arms: int = 24,
    n_arm_types: int = 3,
    n_extra_features: int = 5,
    extra_feature_density: float = 0.5,
    seed: int = 0,
) -> RMABDataset:
    """Toy 2-state / 2-action RMAB with ``n_arm_types`` distinct arm classes.

    Each arm carries a one-hot type prefix in its feature vector followed
    by random binary nuisance traits. Type *k* has dynamics such that
    ``P(next_state = 1 | state = 0, action = 1)`` grows linearly with *k*,
    so the budget-optimal policy concentrates pulls on the highest-type
    arms.

    Args:
        n_arms: number of arms; must be divisible by ``n_arm_types``.
        n_arm_types: number of distinct arm classes.
        n_extra_features: number of nuisance binary features per arm.
        extra_feature_density: Bernoulli probability for each nuisance bit.
        seed: numpy seed.
    """
    if n_arm_types < 1:
        raise ValueError(f"n_arm_types must be >= 1; got {n_arm_types}")
    if n_arms % n_arm_types != 0:
        raise ValueError(f"n_arms ({n_arms}) must be divisible by n_arm_types ({n_arm_types})")

    rng = np.random.default_rng(seed)
    types = np.arange(n_arms) % n_arm_types
    type_onehot = np.eye(n_arm_types, dtype=np.float32)[types]
    extras = (rng.random((n_arms, n_extra_features)) < extra_feature_density).astype(np.float32)
    features = np.concatenate([type_onehot, extras], axis=1)
    feature_names = tuple(
        [f"type_{k}" for k in range(n_arm_types)]
        + [f"trait_{j}" for j in range(n_extra_features)]
    )
    span = max(1, n_arm_types - 1)
    pull_probs = [0.3 + 0.6 * (k / span) for k in range(n_arm_types)]
    feature_descriptions = tuple(
        [
            f"Binary indicator for arm type {k} (P(state 0 → state 1 | pull) = {p:.2f})"
            for k, p in enumerate(pull_probs)
        ]
        + [
            "Binary auxiliary feature unrelated to arm type or reward"
            for _ in range(n_extra_features)
        ]
    )

    T = np.zeros((n_arms, 2, 2, 2), dtype=np.float64)
    for i, k in enumerate(types):
        improvement = pull_probs[k]
        T[i, 0, 0] = [0.7, 0.3]
        T[i, 1, 0] = [0.5, 0.5]
        T[i, 0, 1] = [1.0 - improvement, improvement]
        T[i, 1, 1] = [0.1, 0.9]

    feature_groups: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("Arm Type", tuple(range(n_arm_types))),
    )
    if n_extra_features > 0:
        feature_groups = (
            *feature_groups,
            (
                "Auxiliary Traits",
                tuple(range(n_arm_types, n_arm_types + n_extra_features)),
            ),
        )

    return RMABDataset(
        transition_matrices=T,
        features=features,
        feature_names=feature_names,
        feature_descriptions=feature_descriptions,
        feature_groups=feature_groups,
    )
