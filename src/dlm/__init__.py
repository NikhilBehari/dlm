"""Decision-Language Models for restless multi-armed bandits."""

from .env import Arm, RewardFn, RMABDataset, RMABEnv, binary_arm, from_arms, synthetic_dataset
from .llm import (
    AnthropicProvider,
    EchoProvider,
    GeminiProvider,
    LLMProvider,
    OpenAIProvider,
    load_provider,
)
from .output import default_output_dir
from .rewards import ScriptedReward, compile_reward, is_valid_expression
from .rl import Trainer, TrainerConfig
from .search import (
    CandidateResult,
    DLMConfig,
    DLMTask,
    EvaluationResult,
    PromptConfig,
    StageResult,
    TaskResult,
    evaluate,
    run_task,
)

__version__ = "0.1.0"

__all__ = [
    "RMABDataset", "RMABEnv", "RewardFn", "synthetic_dataset",
    "Arm", "from_arms", "binary_arm",
    "Trainer", "TrainerConfig",
    "ScriptedReward", "compile_reward", "is_valid_expression",
    "LLMProvider", "EchoProvider", "OpenAIProvider", "AnthropicProvider", "GeminiProvider",
    "load_provider",
    "default_output_dir",
    "EvaluationResult", "evaluate",
    "DLMTask", "DLMConfig", "PromptConfig",
    "CandidateResult", "StageResult", "TaskResult", "run_task",
]
