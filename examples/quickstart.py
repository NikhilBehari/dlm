"""Train a Lagrangian-PPO policy on a synthetic 12-arm RMAB."""

from __future__ import annotations

from dlm import (
    RMABEnv,
    ScriptedReward,
    Trainer,
    TrainerConfig,
    evaluate,
    synthetic_dataset,
)


def main() -> None:
    dataset = synthetic_dataset(n_arms=12, n_arm_types=3, seed=0)
    candidate = ScriptedReward("state * agent_feats[2]")
    env = RMABEnv(dataset, budget=4.0, reward_fn=candidate, seed=0)

    trainer = Trainer(env, TrainerConfig(epochs=30, steps_per_epoch=200, seed=0))
    trainer.train(
        callback=lambda i, info: print(
            f"epoch {i:3d}  reward={info['mean_reward']:.3f}"
            f"  cost/step={info['mean_cost_per_step']:.2f}"
            f"  lambda={info['lambda']:.3f}"
        )
    )

    target = ScriptedReward("state * agent_feats[2]")
    result = evaluate(trainer, target, horizon=200, seed=1)
    print("\nfinal eval:")
    print(" ", result.summary())
    print("  pulls per arm:", result.pull_frequency.round(2).tolist())


if __name__ == "__main__":
    main()
