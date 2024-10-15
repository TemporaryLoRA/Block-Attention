import gc
import json
import sys
from tqdm import tqdm
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from transformers import AutoTokenizer, PreTrainedTokenizer

from typing import Dict, List, Optional, Literal, TypedDict

# @dataclass
# class SFTDataInstance:
#     prompt: str
#     answers: List[str]
#     generated: str
#     inputs:

SFTDataInstanceInputs = TypedDict("SFTDataInstanceInputs", {
    "input_ids": List[int],
    "labels": List[int]
})

SFTDataInstance = TypedDict("SFTDataInstance", {
    "prompt": str,
    "answers": List[str],
    "generated": str,
    "inputs": SFTDataInstanceInputs
})


def make_block_attention_for_llama3(ins: SFTDataInstance, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    content = ins["prompt"] + ins["generated"]
    blocks: List[str] = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's questions.\n\n"
    ]
    assert content.startswith(blocks[0])
    content = content[len(blocks[0]):]

    pos = content.find("<|eot_id|>") + len("<|eot_id|>")
    documents = content[:pos]
    instruction_ans_response = content[pos:]

    pos = documents.find("\n- Title:")
    while pos != -1:
        doc = documents[:pos + 1]
        blocks.append(doc)
        documents = documents[pos + 1:]
        pos = documents.find("\n- Title:")
    assert documents.startswith("- Title:") and documents.endswith("<|eot_id|>"), documents
    blocks.append(documents[:-len("<|eot_id|>")])
    blocks.append("<|eot_id|>" + instruction_ans_response)

    assert blocks[-1].startswith(
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Please write a high-quantify answer for the given question using only the provided search documents"
    )
    blocks = [b for b in blocks if b != ""]

    block_input_ids = []
    block_num_tokens = []

    for b in blocks:
        ids = tokenizer.encode(b, add_special_tokens=False)
        block_input_ids.extend(ids)
        block_num_tokens.append(len(ids))

    block_input_ids = torch.tensor(data=block_input_ids, dtype=torch.int64)
    # truth_input_ids = torch.tensor(data=ins["inputs"]["input_ids"], dtype=torch.int64)
    # assert torch.all(input=torch.eq(input=block_input_ids, other=truth_input_ids,))

    total_tokens = block_input_ids.size(-1)
    attention_mask = torch.zeros(size=[total_tokens, total_tokens], dtype=torch.bool)
    helper_mask = torch.tril(input=torch.ones(size=[total_tokens, total_tokens], dtype=torch.bool), diagonal=0)

    offset = 0
    for num_tokens in block_num_tokens[:-1]:
        attention_mask[offset: offset + num_tokens, offset: offset + num_tokens] = helper_mask[:num_tokens, :num_tokens]
        offset += num_tokens
    num_tokens = block_num_tokens[-1]
    attention_mask[-num_tokens:] = helper_mask[total_tokens - num_tokens: total_tokens, :total_tokens]

    attention_mask = attention_mask.to(dtype=torch.bfloat16)
    attention_mask = 1.0 - attention_mask
    attention_mask = attention_mask.masked_fill_(
        mask=attention_mask.to(dtype=torch.bool), value=torch.finfo(torch.bfloat16).min
    )
    return attention_mask.unsqueeze(dim=0).unsqueeze(dim=0)


class BlockAttentionRawDataset:
    def __init__(self, fp: str, tokenizer: PreTrainedTokenizer, max_length: int):
        self.fp = fp
        self.tokenizer = tokenizer
        self.raw_dataset: List[SFTDataInstance] = []
        self.max_length = max_length

    def load_dataset(self):
        self.raw_dataset.clear()
        with open(self.fp, "r", encoding="utf-8") as f:
            items: List[SFTDataInstance] = [json.loads(i) for i in f]

            for i in range(0, len(items)):
                prompt, generated = items[i]["prompt"], items[i]["generated"]
                p_input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                g_input_ids = self.tokenizer.encode(generated, add_special_tokens=False)
                items[i]["inputs"]["input_ids"] = p_input_ids + g_input_ids
                items[i]["inputs"]["labels"] = [-100] * len(p_input_ids) + g_input_ids

            self.raw_dataset.extend([i for i in items if len(i['inputs']['input_ids']) < self.max_length])


class BlockAttentionDataset(Dataset):
    def __init__(self, dataset: BlockAttentionRawDataset, train_prompt: bool):
        super().__init__()
        self.dataset = dataset
        self.gc_counter = 0
        self.train_prompt = train_prompt

    def __len__(self) -> int:
        return len(self.dataset.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self.gc_counter += 1
        if self.gc_counter % 256 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        inputs: SFTDataInstanceInputs = self.dataset.raw_dataset[idx]['inputs']
        attention_mask = make_block_attention_for_llama3(
            ins=self.dataset.raw_dataset[idx], tokenizer=self.dataset.tokenizer
        )
        return {
            "attention_mask": attention_mask,
            "input_ids": torch.tensor(data=[inputs["input_ids"]], dtype=torch.int64),
            "labels": torch.tensor(
                data=[inputs["labels"] if not self.train_prompt else inputs["input_ids"]], dtype=torch.int64
            )
        }


def get_dataset(fp: str, tokenizer: PreTrainedTokenizer, max_length: int,train_prompt: bool) -> BlockAttentionDataset:
    dataset= BlockAttentionRawDataset(fp=fp, tokenizer=tokenizer, max_length=max_length)
    dataset.load_dataset()
    return BlockAttentionDataset(dataset=dataset, train_prompt=train_prompt)



