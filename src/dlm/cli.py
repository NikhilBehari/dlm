"""Command-line entry point for DLM."""

from __future__ import annotations

import argparse
import sys

from .env import RMABEnv, synthetic_dataset
from .llm import EchoProvider
from .rewards import ScriptedReward, compile_reward
from .rl import Trainer, TrainerConfig
from .search import DLMConfig, DLMTask, run_task


def _cmd_train(args: argparse.Namespace) -> int:
    dataset = synthetic_dataset(n_arms=args.n_arms, seed=args.seed)
    reward = ScriptedReward(args.reward)
    env = RMABEnv(dataset, budget=args.budget, reward_fn=reward, seed=args.seed)
    trainer = Trainer(env, TrainerConfig(epochs=args.epochs, seed=args.seed))
    history = trainer.train(
        callback=lambda i, info: print(
            f"epoch {i:3d}  reward={info['mean_reward']:.3f}  "
            f"cost/step={info['mean_cost_per_step']:.2f}  "
            f"lambda={info['lambda']:.3f}"
        )
    )
    final = history[-1]
    print(f"\nfinal reward={final['mean_reward']:.3f}, lambda={final['lambda']:.3f}")
    return 0


def _cmd_dlm(args: argparse.Namespace) -> int:
    dataset = synthetic_dataset(n_arms=args.n_arms, seed=args.seed)
    target = compile_reward(args.target)
    task = DLMTask(name="synthetic", goal=args.goal, target_reward=target)
    provider = EchoProvider(rewards=args.candidate, picks=[0])
    result = run_task(
        task,
        dataset,
        budget=args.budget,
        provider=provider,
        trainer_config=TrainerConfig(epochs=args.epochs, seed=args.seed),
        dlm_config=DLMConfig(num_stages=args.stages, num_candidates=len(args.candidate)),
        progress=print,
    )
    best = result.best_candidate
    print(f"\nbest expression: {best.source}")
    print(best.evaluation.summary())
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dlm", description="Decision-Language Model CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a single policy on toy data with a fixed reward.")
    p_train.add_argument("--reward", default="state * agent_feats[0]")
    p_train.add_argument("--n-arms", type=int, default=12)
    p_train.add_argument("--budget", type=float, default=4.0)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.set_defaults(func=_cmd_train)

    p_dlm = sub.add_parser("dlm", help="Run a DLM cycle with EchoProvider on toy data.")
    p_dlm.add_argument("--goal", default="prioritize type-2 arms")
    p_dlm.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="A candidate reward expression. Pass --candidate multiple times.",
    )
    p_dlm.add_argument("--target", default="state * agent_feats[2]")
    p_dlm.add_argument("--n-arms", type=int, default=12)
    p_dlm.add_argument("--budget", type=float, default=4.0)
    p_dlm.add_argument("--epochs", type=int, default=20)
    p_dlm.add_argument("--stages", type=int, default=1)
    p_dlm.add_argument("--seed", type=int, default=0)
    p_dlm.set_defaults(func=_cmd_dlm)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "dlm" and not args.candidate:
        args.candidate = ["state * agent_feats[2]", "state * (agent_feats[1] or agent_feats[2])"]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
