import sys

sys.path.append(".")

from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (
    Trainer, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM, TrainingArguments,
    DataCollatorForSeq2Seq, HfArgumentParser, TrainerState, TrainerControl, TrainerCallback
)

from typing import List

from src.data.sft import get_dataset
from src.args import ModelArgs, DataArgs


class CustomSave(TrainerCallback):

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save = False


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


class SingleCaseCollator:
    def __init__(self):
        pass

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        assert len(features) == 1
        features[0]["input_ids"] = features[0]["input_ids"].unsqueeze(dim=0)
        features[0]["labels"] = features[0]["labels"].unsqueeze(dim=0)
        return features[0]


def main():
    parser = HfArgumentParser((TrainingArguments, ModelArgs, DataArgs))
    train_args, model_args, data_args = parser.parse_args_into_dataclasses()  # type: TrainingArguments, ModelArgs, DataArgs

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
        attn_implementation="flash_attention_2",
    )

    train_dataset: Dataset = get_dataset(
        fp=data_args.train_fp, max_length=data_args.max_length, tokenizer=tokenizer
    )
    eval_dataset: Dataset = get_dataset(
        fp=data_args.eval_fp, max_length=data_args.max_length, tokenizer=tokenizer
    )

    model.train()
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt"
        ) if train_args.per_device_train_batch_size > 1 else SingleCaseCollator(),
        callbacks=[CustomSave()],
        compute_loss_func=CEReductionSumOutputLoss(
            vocab_size=model.config.vocab_size
        ) if model_args.loss_reduction == "sum" else None,
    )
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)


if __name__ == '__main__':
    main()
