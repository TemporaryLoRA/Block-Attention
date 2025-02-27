import os
import sys

sys.path.append(".")

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (
    Trainer, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM, TrainingArguments,
    HfArgumentParser, TrainerState, TrainerControl, TrainerCallback
)

from src.args import ModelArgs, DataArgs, BlockArgs
from src.data.block import get_dataset, BlockBatchCollector


class CustomSave(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save = True


class CEReductionSumOutputLoss:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(
            self, outputs: CausalLMOutputWithPast, labels, num_items_in_batch: int = None, ignore_index: int = -100,
            **kwargs
    ):
        logits = outputs.logits.float()
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index, reduction="sum")
        return loss


def main():
    parser = HfArgumentParser((TrainingArguments, ModelArgs, DataArgs, BlockArgs))
    train_args, model_args, data_args, block_args = parser.parse_args_into_dataclasses()  # type: TrainingArguments, ModelArgs, DataArgs, BlockArgs
    train_args.remove_unused_columns = False
    print("remove unused columns: ", train_args.remove_unused_columns)
    print("save_only_model: ", train_args.save_only_model)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name,
        trust_remote_code=False,
        use_cache=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    train_dataset: Dataset = get_dataset(
        fp=data_args.train_fp,
        model_name=model_args.model_name,
        max_length=data_args.max_length,
        tokenizer=tokenizer,
        train_prompt=block_args.train_prompt == 'true',
        train_full_attention=block_args.train_full_attention == 'true',
        add_special_domain_tokens=block_args.add_special_domain_tokens == 'true'
    )
    eval_dataset: Dataset = get_dataset(
        fp=data_args.eval_fp,
        model_name=model_args.model_name,
        max_length=data_args.max_length,
        tokenizer=tokenizer,
        train_prompt=False,
        train_full_attention=True,
        add_special_domain_tokens=block_args.add_special_domain_tokens == 'true'
    )

    model.train()
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=BlockBatchCollector(pad_token_id=tokenizer.pad_token_id, max_length=data_args.max_length),
        callbacks=[CustomSave()],
        compute_loss_func=CEReductionSumOutputLoss(
            vocab_size=model.config.vocab_size
        ) if model_args.loss_reduction == "sum" else None,
    )
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)


if __name__ == '__main__':
    main()
