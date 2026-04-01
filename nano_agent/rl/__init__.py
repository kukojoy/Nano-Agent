"""模型后训练模块

本模块提供基于 TRL 的模型后训练功能, 包括:
- SFT
- GRPO
"""

# 检查 TRL 是否可用
try:
    import trl
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

from .trainers import SFTTrainerWrapper, GRPOTrainerWrapper
from .rl_datasets import (
    GSM8KDataset,
    create_math_dataset,
    create_sft_dataset,
    create_rl_dataset,
    preview_dataset
)
from .rewards import (
    MathRewardFunction,
    create_accuracy_reward,
    create_length_penalty_reward,
    create_step_reward,
    evaluate_rewards
)
from .utils import TrainingConfig, setup_training_environment


__all__ = [
    # 可用性标志
    "TRL_AVAILABLE",

    # 训练器
    "SFTTrainerWrapper",
    "GRPOTrainerWrapper",
    "PPOTrainerWrapper",

    # 数据集
    "GSM8KDataset",
    "create_math_dataset",
    "create_sft_dataset",
    "create_rl_dataset",
    "preview_dataset",
    "format_math_dataset",

    # 奖励函数
    "MathRewardFunction",
    "create_accuracy_reward",
    "create_length_penalty_reward",
    "create_step_reward",
    "evaluate_rewards",

    # 工具函数
    "TrainingConfig",
    "setup_training_environment",
]