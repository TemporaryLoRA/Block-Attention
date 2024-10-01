from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Literal


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


