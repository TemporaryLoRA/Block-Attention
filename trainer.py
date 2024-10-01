import os
import sys

sys.path.append(".")

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (
    Trainer, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM, TrainingArguments,
    DataCollatorForSeq2Seq, HfArgumentParser, TrainerState, TrainerControl, TrainerCallback, AutoConfig, LlamaConfig
)

from typing import Any, Dict, List, TypedDict, Union, Tuple, Optional

from src.data.sft import get_dataset as get_dataset_sft
from src.data.block import get_dataset as get_dataset_block

from src.args import ModelArgs, DataArgs


def _collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_length = max([f["input_ids"].size(-1) for f in features])
    for f in features:
        pad_length = max_length - f["input_ids"].size(-1)
        if pad_length != 0:
            f["input_ids"] = F.pad(
                input=f["input_ids"], pad=[0, pad_length], mode="constant", value=tokenizer.pad_token_id
            )
            f["labels"] = F.pad(input=f["labels"], pad=[0, pad_length], mode="constant", value=-100)
            f["attention_mask"] = F.pad(
                input=f["attention_mask"], pad=[0, pad_length, 0, pad_length], mode="constant",
                value=torch.finfo(f["attention_mask"]).min
            )
    batch = {}
    for k in {"input_ids", "attention_mask", "labels"}:
        tensors = [f[k] for f in features]
        batch[k] = torch.cat(tensors=tensors, dim=0)
    return batch


def data_collator(features: List[Dict[str, Any]], return_tensors: Optional[bool] = False) -> Dict[str, Any]:
    if len(features) == 0:
        return features[0]
    return _collator(features=features)

if __name__ == '__main__':
    parser = HfArgumentParser([TrainingArguments, ModelArgs, DataArgs])
    train_args, model_args, data_args = parser.parse_args_into_dataclasses()
    train_args: TrainingArguments
    model_args: ModelArgs
    data_args: DataArgs

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name,
        trust_remote_code=False,
        use_cache=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if data_args.train_method == "sft" else "sdpa"
    )

    with train_args.main_process_first(desc="Load dataset: ", local=True):
        if data_args.train_method == "sft":
            train_dataset: Dataset = get_dataset_sft(fp=data_args.train_fp, max_length=data_args.max_length)
            eval_dataset: Dataset = get_dataset_sft(fp=data_args.eval_fp, max_length=data_args.max_length)
        else:
            train_dataset: Dataset = get_dataset_block(
                fp=data_args.train_fp, max_length=data_args.max_length, tokenizer=tokenizer,
                train_prompt=data_args.train_prompt
            )
            eval_dataset: Dataset = get_dataset_block(
                fp=data_args.train_fp, max_length=data_args.max_length, tokenizer=tokenizer,
                train_prompt=data_args.train_prompt
            )

    model.train()
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, max_length=data_args.max_length, pad_to_multiple_of=8,
            label_pad_token_id=-100, return_tensors="pt"
        ) if data_args.train_method == "sft" else data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    trainer.train()
