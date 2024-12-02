"""Construct an :class:`~dlm.RMABDataset` from your own arrays."""

from __future__ import annotations

import numpy as np

from dlm import (
    Arm,
    RMABEnv,
    ScriptedReward,
    Trainer,
    TrainerConfig,
    binary_arm,
    evaluate,
    from_arms,
)


def main() -> None:
    rng = np.random.default_rng(42)

    # Each arm has a single continuous feature ("potential") that drives the
    # probability of transitioning to state 1 under an active pull.
    n_arms = 6
    potentials = rng.random(n_arms).astype(np.float32)
    arms: list[Arm] = [
        binary_arm(
            features=[float(p)],
            p_pull_at_0=0.2 + 0.7 * float(p),
            p_pull_at_1=0.9,
            p_pass_at_0=0.3,
            p_pass_at_1=0.5,
        )
        for p in potentials
    ]
    dataset = from_arms(arms, feature_names=("potential",))

    reward = ScriptedReward("state * agent_feats[0]")
    env = RMABEnv(dataset, budget=2.0, reward_fn=reward, seed=0)

    trainer = Trainer(
        env,
        TrainerConfig(
            epochs=40,
            steps_per_epoch=200,
            lambda_lr=1e-2,
            lambda_update_every=2,
            seed=0,
        ),
    )
    trainer.train()
    result = evaluate(trainer, reward, horizon=200, seed=1)
    print(result.summary())
    print("pulls per arm:", result.pull_frequency.round(2).tolist())


if __name__ == "__main__":
    main()
