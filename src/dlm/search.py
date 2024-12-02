"""Evolutionary outer loop and policy evaluation under a target reward."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .env import RewardFn, RMABDataset, RMABEnv
from .llm import LLMProvider
from .llm.base import _is_rate_limit
from .output import NullWriter, OutputWriter
from .rewards import ScriptedReward, is_valid_expression
from .rl import Trainer, TrainerConfig


@dataclass
class EvaluationResult:
    """Summary of a deterministic policy rollout under a target reward.

    ``reward_per_feature`` stores raw reward attribution per feature: the
    sum of ``feature_value * reward`` over the rollout. For binary
    features this equals the total reward earned by arms with the feature
    active; for continuous features the magnitude is weighted by the
    feature value. Percentages are derived against :attr:`total_reward`
    when the dict is rendered.
    """

    total_reward: float
    mean_reward_per_step: float
    pull_frequency: np.ndarray
    reward_per_feature: dict[str, float]

    def summary(self, top_k: int = 5) -> str:
        ranked = sorted(self.reward_per_feature.items(), key=lambda kv: -kv[1])[:top_k]
        total = self.total_reward if self.total_reward > 0 else 1.0
        feats = ", ".join(f"{k}={v / total * 100:.2f}%" for k, v in ranked)
        return (
            f"total={self.total_reward:.2f}  "
            f"per-step={self.mean_reward_per_step:.4f}  "
            f"top-{top_k} features: {feats}"
        )


def evaluate(
    trainer: Trainer,
    target_reward: RewardFn,
    horizon: int = 200,
    seed: int = 0,
) -> EvaluationResult:
    """Roll out the trainer's deterministic policy under ``target_reward``."""
    dataset = trainer.env.dataset
    env = RMABEnv(dataset, trainer.env.budget, target_reward, seed=seed)
    state = env.reset()

    total = 0.0
    pulls = np.zeros(dataset.n_arms, dtype=np.float64)
    feat_sum = np.zeros(dataset.n_features, dtype=np.float64)
    feature_weights = dataset.features.astype(np.float64)
    for _ in range(horizon):
        action = trainer.act(state, deterministic=True)
        pulls += (action > 0).astype(np.float64)
        state, rewards, _, _ = env.step(action)
        total += float(rewards.sum())
        feat_sum += feature_weights.T @ rewards.astype(np.float64)

    names = dataset.feature_names or tuple(f"feature_{j}" for j in range(dataset.n_features))
    reward_per_feature = {names[j]: float(feat_sum[j]) for j in range(dataset.n_features)}
    return EvaluationResult(
        total_reward=total,
        mean_reward_per_step=total / max(1, horizon),
        pull_frequency=pulls / max(1, horizon),
        reward_per_feature=reward_per_feature,
    )


@dataclass
class DLMTask:
    """A reward-design task: a natural-language goal and an optional target reward.

    When ``target_reward`` is supplied it is used as the ground-truth scoring
    function during evaluation, never shown to the LLM. When ``None``, each
    candidate is evaluated under its own proposed reward; this is the right
    setting for users who only want to *generate* reward functions without a
    pre-defined oracle.
    """

    name: str
    goal: str
    target_reward: RewardFn | None = None


DEFAULT_DOMAIN_CONTEXT = "a restless multi-armed bandit resource-allocation problem"
DEFAULT_EXAMPLE_GOAL = "While prioritizing all, emphasize arms whose feature 0 is set."
DEFAULT_EXAMPLE_EXPRESSION = "state * 0.1 + 2 * state * agent_feats[0]"
DEFAULT_EXAMPLE_REASONING = (
    "Feature 0 corresponds to agent_feats[0], so we add a bonus weighted by "
    "agent_feats[0]. The base term `state * 0.1` keeps reward positive and "
    "monotonically increasing with state."
)


@dataclass
class PromptConfig:
    """User-configurable prompt elements shown to the LLM.

    Every field is optional. When omitted, the package falls back to the
    generic defaults declared at the top of this module.
    """

    domain_context: str | None = None
    example_goal: str | None = None
    example_expression: str | None = None
    example_reasoning: str | None = None
    chain_of_thought: bool = False
    request_explanation: bool = False


@dataclass
class DLMConfig:
    """Outer-loop hyperparameters."""

    num_stages: int = 2
    num_candidates: int = 2
    max_propose_retries: int = 10
    eval_horizon: int = 200
    seed: int = 0
    prompt: PromptConfig = field(default_factory=PromptConfig)


@dataclass
class CandidateResult:
    """One proposed reward expression with its training and evaluation outcome."""

    source: str
    reward_fn: ScriptedReward
    evaluation: EvaluationResult
    explanation: str | None = None


@dataclass
class StageResult:
    """All candidates within one stage and the provider's pick."""

    candidates: list[CandidateResult]
    best_index: int


@dataclass
class TaskResult:
    """Full result for one task across all stages."""

    task: DLMTask
    stages: list[StageResult] = field(default_factory=list)

    @property
    def best_candidate(self) -> CandidateResult:
        last = self.stages[-1]
        return last.candidates[last.best_index]


def _format_feature_table(dataset: RMABDataset) -> str:
    if not dataset.feature_names:
        return "  (no feature names provided)"
    lines = ["Index. Name - Description"]
    for i, name in enumerate(dataset.feature_names):
        line = f"  {i}. {name}"
        if dataset.feature_descriptions:
            desc = dataset.feature_descriptions[i]
            if desc:
                line += f" - {desc}"
        lines.append(line)
    return "\n".join(lines)


def _format_reflection(dataset: RMABDataset, evaluation: EvaluationResult) -> str:
    """Render the per-feature reward distribution as percentages of total reward.

    Each feature's percentage is ``feature_attribution / total_reward * 100``,
    matching the reflection format used in the DLM paper. With
    ``dataset.feature_groups`` declared, features are grouped by category
    so the LLM sees a structured breakdown.
    """
    names = dataset.feature_names or tuple(
        f"feature_{j}" for j in range(dataset.n_features)
    )
    rpf = evaluation.reward_per_feature
    total = evaluation.total_reward if evaluation.total_reward > 0 else 1.0
    if dataset.feature_groups:
        lines: list[str] = []
        for group_name, idxs in dataset.feature_groups:
            lines.append(f"Category: {group_name}")
            for j in idxs:
                pct = rpf.get(names[j], 0.0) / total * 100
                lines.append(f"  {names[j]}: {pct:.2f}%")
        return "\n".join(lines)
    ranked = sorted(rpf.items(), key=lambda kv: -kv[1])
    return "\n".join(f"  {k}: {v / total * 100:.2f}%" for k, v in ranked)


def _propose_prompt(
    goal: str,
    feature_table: str,
    n_features: int,
    n_states: int,
    history: list[dict[str, str | None]],
    prompt_cfg: PromptConfig,
) -> str:
    """Render the reward-proposal prompt, matching the structure used in the DLM paper."""
    domain = prompt_cfg.domain_context or DEFAULT_DOMAIN_CONTEXT
    goal_clause = goal.rstrip(". ")
    ex_goal = (prompt_cfg.example_goal or DEFAULT_EXAMPLE_GOAL).rstrip(". ")
    ex_expr = prompt_cfg.example_expression or DEFAULT_EXAMPLE_EXPRESSION
    ex_reasoning = prompt_cfg.example_reasoning or DEFAULT_EXAMPLE_REASONING

    task_lines = [
        "Your task:",
        "1. Write a single-line Python reward expression. Do not use the word "
        "`return` and do not import any libraries. Wrap your expression in "
        "triple $ signs: $$$[YOUR FUNCTION]$$$.",
    ]
    if prompt_cfg.request_explanation:
        task_lines.append(
            "2. Provide a one-sentence explanation of how the expression aligns "
            "with the goal. Wrap it in triple % signs: %%%[YOUR EXPLANATION]%%%."
        )
    task_lines.append(
        "Reward must be positive and strictly increasing in `state`. Use "
        "`and` / `or` rather than the bitwise `&` / `|`."
    )
    task_block = "\n".join(task_lines)

    example_lines = [
        "Example",
        f"  Goal     : {ex_goal}",
    ]
    if prompt_cfg.chain_of_thought:
        example_lines.append(
            f"  Reasoning: Let's think about this step by step. {ex_reasoning}"
        )
    example_lines.append(f"  Response : $$${ex_expr}$$$")
    if prompt_cfg.request_explanation:
        example_lines.append(f"  Rationale: %%%{ex_reasoning}%%%")
    example_text = "\n".join(example_lines)

    history_lines = []
    for entry in history:
        line = f"  - {entry['expression']}"
        if entry.get("explanation"):
            line += f"  ({entry['explanation']})"
        history_lines.append(line)
    history_text = "\n".join(history_lines) or "  (none yet)"

    state_range = "0, 1" if n_states == 2 else f"0, 1, ..., {n_states - 1}"
    return (
        f"Create a Python reward expression for {domain}. The objective is to "
        f"prioritize higher states while pursuing the following goal:\n"
        f"  {goal_clause}\n\n"
        f"The expression must use `state` (an integer in {{{state_range}}}) and "
        f"`agent_feats` (a length-{n_features} feature array) to direct the "
        f"RL agent.\n\n"
        f"Features available:\n{feature_table}\n\n"
        f"{task_block}\n\n"
        f"{example_text}\n\n"
        f"Propose a fresh reward expression for the goal above. Previous "
        f"attempts:\n{history_text}"
    )


def _select_prompt(
    goal: str,
    feature_table: str,
    candidates: list[CandidateResult],
    dataset: RMABDataset,
    prompt_cfg: PromptConfig,
    has_target: bool,
) -> str:
    """Render the reflection / selection prompt, matching the DLM paper's format."""
    domain = prompt_cfg.domain_context or DEFAULT_DOMAIN_CONTEXT
    goal_clause = goal.rstrip(". ")
    intro = (
        "I tried several reward expressions. Below is each candidate followed by "
        f"the distribution of reward it produced across {dataset.n_features} agent "
        "features under the held-out target reward."
        if has_target
        else (
            "I tried several reward expressions. Below is each candidate followed by "
            f"the distribution of reward it produced across {dataset.n_features} "
            "agent features under its own reward. Compare the per-feature allocation "
            "patterns rather than absolute totals (each candidate is measured on its "
            "own reward scale)."
        )
    )
    parts = [
        f"My goal was to create a Python reward expression for {domain} with the "
        f"following objective:",
        f"  {goal_clause}",
        "",
        intro,
        "",
        "Features:",
        feature_table,
        "",
        "Candidates:",
    ]
    for i, c in enumerate(candidates):
        parts.append(f"\nIndex {i}")
        parts.append(f"  Reward Function: {c.source}")
        if c.explanation:
            parts.append(f"  Rationale: {c.explanation}")
        parts.append("  Reflection:")
        parts.append(_format_reflection(dataset, c.evaluation))
    parts.append("")
    parts.append(
        "Based on the reward distributions and the goal above, identify the index "
        "of the most effective reward expression."
    )
    if prompt_cfg.request_explanation:
        parts.append(
            "Respond in the format `The best reward function is at index: [INDEX]`, "
            "then add a one-line %%%...%%% explanation of the choice."
        )
    else:
        parts.append(
            "Respond EXACTLY in the format `The best reward function is at "
            "index: [INDEX]`."
        )
    return "\n".join(parts)


def _extract_expression(text: str) -> str:
    if "$$$" in text:
        chunks = text.split("$$$")
        if len(chunks) >= 3:
            return chunks[-2].strip()
    return text.strip()


def _extract_explanation(text: str) -> str | None:
    if "%%%" in text:
        chunks = text.split("%%%")
        if len(chunks) >= 3:
            return chunks[-2].strip() or None
    return None


def _propose_valid_expression(
    provider: LLMProvider,
    prompt: str,
    n_features: int,
    max_retries: int,
) -> tuple[str, str | None, str]:
    """Return ``(expression, explanation, raw_response)``.

    On a rate-limit or quota signal from the provider, raises
    :class:`RuntimeError` immediately without spending further retries.
    Other transient provider errors are retried up to ``max_retries``.
    """
    last_error = "no proposals attempted"
    for _ in range(max_retries):
        try:
            text = provider.propose_reward(prompt)
        except Exception as e:
            if _is_rate_limit(e):
                raise RuntimeError(f"LLM provider rate-limit or quota exceeded: {e}") from e
            last_error = f"provider raised: {e!r}"
            continue
        expr = _extract_expression(text)
        if is_valid_expression(expr, n_features=n_features):
            return expr, _extract_explanation(text), text
        last_error = f"invalid expression: {expr!r}"
    raise RuntimeError(
        f"could not obtain a valid reward expression after {max_retries} tries ({last_error})"
    )


def _eval_dict(e: EvaluationResult) -> dict[str, Any]:
    return {
        "total_reward": e.total_reward,
        "mean_reward_per_step": e.mean_reward_per_step,
        "pull_frequency": e.pull_frequency.tolist(),
        "reward_per_feature": e.reward_per_feature,
    }


def _result_dict(result: TaskResult, aborted: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": {"name": result.task.name, "goal": result.task.goal},
        "stages": [
            {
                "best_index": s.best_index,
                "candidates": [
                    {
                        "source": c.source,
                        "explanation": c.explanation,
                        "evaluation": _eval_dict(c.evaluation),
                    }
                    for c in s.candidates
                ],
            }
            for s in result.stages
        ],
    }
    if result.stages:
        best = result.best_candidate
        payload["best_expression"] = best.source
        payload["best_explanation"] = best.explanation
        payload["best_evaluation"] = _eval_dict(best.evaluation)
    if aborted is not None:
        payload["aborted"] = aborted
    return payload


def _config_dict(
    task: DLMTask,
    budget: float,
    dlm_cfg: DLMConfig,
    trainer_cfg: TrainerConfig,
) -> dict[str, Any]:
    target_repr: str | None
    if task.target_reward is None:
        target_repr = None
    else:
        target_repr = getattr(task.target_reward, "source", "<callable>")
    return {
        "task": {"name": task.name, "goal": task.goal, "target_reward": target_repr},
        "budget": budget,
        "dlm_config": dataclasses.asdict(dlm_cfg),
        "trainer_config": dataclasses.asdict(trainer_cfg),
    }


def _fenced(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def _transcript_header(task: DLMTask, budget: float, cfg: DLMConfig) -> str:
    return (
        f"# DLM run: {task.name}\n\n"
        f"**Goal:** {task.goal}  \n"
        f"**Budget:** {budget}  \n"
        f"**Stages × Candidates:** {cfg.num_stages} × {cfg.num_candidates}\n\n"
    )


def _transcript_candidate(
    cand_idx: int,
    prompt: str,
    response: str,
    expression: str,
    explanation: str | None,
    evaluation: EvaluationResult,
) -> str:
    explanation_md = (
        f"**Explanation:** {explanation}\n\n" if explanation else ""
    )
    return (
        f"### Candidate {cand_idx + 1}\n\n"
        f"**Prompt to LLM:**\n\n{_fenced(prompt)}\n\n"
        f"**LLM response:**\n\n{_fenced(response)}\n\n"
        f"**Extracted expression:** `{expression}`\n\n"
        f"{explanation_md}"
        f"**Evaluation:** {evaluation.summary()}\n\n"
    )


def _transcript_selection(
    prompt: str,
    response: str,
    best_index: int,
    best_source: str,
    explanation: str | None = None,
) -> str:
    explanation_md = f"**Explanation:** {explanation}\n\n" if explanation else ""
    return (
        "### Selection\n\n"
        f"**Prompt to LLM:**\n\n{_fenced(prompt)}\n\n"
        f"**LLM response:**\n\n{_fenced(response)}\n\n"
        f"{explanation_md}"
        f"**Selected:** index {best_index} (`{best_source}`)\n\n"
        "---\n\n"
    )


def _transcript_final(result: TaskResult) -> str:
    best = result.best_candidate
    return (
        "## Final\n\n"
        f"**Best expression:** `{best.source}`  \n"
        f"**Evaluation:** {best.evaluation.summary()}\n"
    )


def run_task(
    task: DLMTask,
    dataset: RMABDataset,
    budget: float,
    provider: LLMProvider,
    trainer_config: TrainerConfig | None = None,
    dlm_config: DLMConfig | None = None,
    progress: Callable[[str], None] | None = None,
    output_dir: str | Path | None = None,
) -> TaskResult:
    """Run the DLM outer loop for a single task.

    When ``output_dir`` is provided, the run is persisted to disk: ``config.json``,
    ``run.log``, ``transcript.md``, ``training.jsonl``, ``prompts.jsonl``,
    ``candidates.jsonl``, ``result.json``. ``progress`` receives the same
    formatted lines that are written to ``run.log``.
    """
    dlm_cfg = dlm_config or DLMConfig()
    tr_cfg = trainer_config or TrainerConfig()
    writer = OutputWriter(output_dir) if output_dir is not None else NullWriter()

    def emit(line: str) -> None:
        if progress is not None:
            progress(line)
        writer.log(line)

    writer.write_config(_config_dict(task, budget, dlm_cfg, tr_cfg))
    writer.transcript(_transcript_header(task, budget, dlm_cfg))
    feature_table = _format_feature_table(dataset)

    emit("=" * 60)
    emit(f"task   : {task.name}")
    emit(f"goal   : {task.goal}")
    emit(f"budget : {budget}")
    emit(f"stages : {dlm_cfg.num_stages} x {dlm_cfg.num_candidates} candidates")
    emit("=" * 60)

    result = TaskResult(task=task)
    history: list[dict[str, str | None]] = []

    try:
        for stage in range(dlm_cfg.num_stages):
            emit("")
            emit(f"[stage {stage + 1}/{dlm_cfg.num_stages}]")
            writer.transcript(f"## Stage {stage + 1} / {dlm_cfg.num_stages}\n\n")
            candidates: list[CandidateResult] = []

            for cand_idx in range(dlm_cfg.num_candidates):
                propose_prompt = _propose_prompt(
                    task.goal,
                    feature_table,
                    dataset.n_features,
                    dataset.n_states,
                    history,
                    dlm_cfg.prompt,
                )
                expr, explanation, raw_response = _propose_valid_expression(
                    provider=provider,
                    prompt=propose_prompt,
                    n_features=dataset.n_features,
                    max_retries=dlm_cfg.max_propose_retries,
                )
                writer.write_prompt(
                    {
                        "stage": stage,
                        "candidate": cand_idx,
                        "kind": "propose",
                        "prompt": propose_prompt,
                        "response": raw_response,
                        "expression": expr,
                        "explanation": explanation,
                    }
                )
                emit(f"  candidate {cand_idx + 1}/{dlm_cfg.num_candidates}: {expr}")
                if explanation:
                    emit(f"    rationale: {explanation}")

                candidate_seed = dlm_cfg.seed + stage * 1000 + cand_idx
                reward = ScriptedReward(expr)
                env = RMABEnv(dataset, budget, reward, seed=candidate_seed)
                cand_tr_cfg = dataclasses.replace(tr_cfg, seed=candidate_seed)
                trainer = Trainer(env, cand_tr_cfg)

                def on_epoch(
                    epoch: int,
                    info: dict[str, Any],
                    _stage: int = stage,
                    _cand: int = cand_idx,
                ) -> None:
                    writer.write_training_epoch(
                        {"stage": _stage, "candidate": _cand, **info}
                    )

                trainer.train(callback=on_epoch)
                eval_reward = task.target_reward if task.target_reward is not None else reward
                evaluation = evaluate(
                    trainer,
                    eval_reward,
                    horizon=dlm_cfg.eval_horizon,
                    seed=dlm_cfg.seed,
                )
                emit(f"    eval: {evaluation.summary()}")
                candidates.append(
                    CandidateResult(
                        source=expr,
                        reward_fn=reward,
                        evaluation=evaluation,
                        explanation=explanation,
                    )
                )
                writer.write_candidate(
                    {
                        "stage": stage,
                        "candidate": cand_idx,
                        "source": expr,
                        "explanation": explanation,
                        "evaluation": _eval_dict(evaluation),
                    }
                )
                writer.transcript(
                    _transcript_candidate(
                        cand_idx, propose_prompt, raw_response, expr, explanation, evaluation
                    )
                )

            select_prompt = _select_prompt(
                task.goal,
                feature_table,
                candidates,
                dataset,
                dlm_cfg.prompt,
                has_target=task.target_reward is not None,
            )
            best, select_response = provider.pick_index(select_prompt, len(candidates))
            best = max(0, min(best, len(candidates) - 1))
            best_source = candidates[best].source
            best_explanation = candidates[best].explanation
            select_explanation = (
                _extract_explanation(select_response)
                if dlm_cfg.prompt.request_explanation
                else None
            )
            writer.write_prompt(
                {
                    "stage": stage,
                    "kind": "select",
                    "prompt": select_prompt,
                    "response": select_response,
                    "best_index": best,
                    "selected_source": best_source,
                    "selected_explanation": best_explanation,
                    "selector_rationale": select_explanation,
                }
            )
            writer.transcript(
                _transcript_selection(
                    select_prompt, select_response, best, best_source, select_explanation
                )
            )
            emit(f"  selected index {best}: {best_source}")
            history.append({"expression": best_source, "explanation": best_explanation})
            result.stages.append(StageResult(candidates=candidates, best_index=best))

        emit("")
        emit("=" * 60)
        emit(f"best expression : {result.best_candidate.source}")
        emit(f"best evaluation : {result.best_candidate.evaluation.summary()}")
        if output_dir is not None:
            emit(f"output saved to : {output_dir}")
        emit("=" * 60)
        writer.transcript(_transcript_final(result))
        writer.write_result(_result_dict(result))
        return result
    except KeyboardInterrupt:
        emit("")
        emit("interrupted; saving partial result")
        writer.write_result(_result_dict(result, aborted="interrupted"))
        raise
    except Exception as exc:
        emit("")
        emit("=" * 60)
        emit(f"aborted: {type(exc).__name__}: {exc}")
        emit(f"completed stages: {len(result.stages)} / {dlm_cfg.num_stages}")
        if output_dir is not None:
            emit(f"partial output saved to : {output_dir}")
        emit("=" * 60)
        writer.write_result(_result_dict(result, aborted=f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        writer.close()
