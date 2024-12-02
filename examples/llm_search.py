"""Drive the DLM outer loop with a hosted model."""

from __future__ import annotations

from dlm import (
    DLMConfig,
    DLMTask,
    TrainerConfig,
    compile_reward,
    default_output_dir,
    load_provider,
    run_task,
    synthetic_dataset,
)


def main() -> None:
    dataset = synthetic_dataset(n_arms=12, n_arm_types=3, seed=0)
    provider = load_provider("llm_config.toml")
    task = DLMTask(
        name="prioritize_type_2",
        goal="Focus the budget on arms whose type is 2.",
        target_reward=compile_reward("state * agent_feats[2]"),
    )
    result = run_task(
        task,
        dataset,
        budget=4.0,
        provider=provider,
        trainer_config=TrainerConfig(epochs=20, steps_per_epoch=200, seed=0),
        dlm_config=DLMConfig(num_stages=2, num_candidates=2),
        progress=print,
        output_dir=default_output_dir(task.name),
    )
    print(f"\nbest expression : {result.best_candidate.source}")
    print(result.best_candidate.evaluation.summary())


if __name__ == "__main__":
    main()
