from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataArgs:
    train_fp: str
    eval_fp: str
    train_method: Literal["sft", "block"]
    train_prompt: bool = field(default=False)
    max_length: int = field(default=8 * 1024)


@dataclass
class ModelArgs:
    model_name: str
    loss_reduction: Literal["mean", "sum"] = field(default="mean")


@dataclass
class BlockArgs:
    # Whether to train the prompt
    train_prompt: Literal["true", "false"]
    # Whether to perform general SFT (Supervised Fine-Tuning) while conducting Block-Attention training
    train_full_attention: Literal["true", "false"]
    # Whether to add the special token `[Block-Attention]` during Block-Attention training
    add_special_domain_tokens: Literal["true", "false"]
