"""DLM outer loop on a multi-feature goal.

Unlike :mod:`llm_search`, the goal here cannot be satisfied by a single
feature: the target reward blends two arm types with different weights,
so the LLM has to combine multiple feature indices in its proposal.
"""

from __future__ import annotations

from dlm import (
    DLMConfig,
    DLMTask,
    PromptConfig,
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
        name="weighted_type1_type2",
        goal=(
            "Allocate most of the budget to arms of type 2, but give partial "
            "credit to arms of type 1 as well, weighted lower than type 2."
        ),
        target_reward=compile_reward(
            "state * (agent_feats[2] + 0.5 * agent_feats[1])"
        ),
    )
    result = run_task(
        task,
        dataset,
        budget=4.0,
        provider=provider,
        trainer_config=TrainerConfig(epochs=20, steps_per_epoch=200, seed=0),
        dlm_config=DLMConfig(
            num_stages=2,
            num_candidates=2,
            prompt=PromptConfig(request_explanation=True),
        ),
        progress=print,
        output_dir=default_output_dir(task.name),
    )
    print(f"\nselected expression : {result.best_candidate.source}")
    print(result.best_candidate.evaluation.summary())


if __name__ == "__main__":
    main()
