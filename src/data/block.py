import gc
import json
import random
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

SFTInputs = TypedDict("SFTInputs", {
    "input_ids": List[int],
    "labels": List[int]
})

SFTInstance = TypedDict("SFTInstance", {
    "inputs": SFTInputs,
    "chunks": List[str],
    "block_inputs": SFTInputs,
    "block_tokens": List[int],
    "response_tokens": int,
    "train_block": bool
})

DatasetOutput = TypedDict("DatasetOutput", {
    "input_ids": torch.Tensor,
    "labels": torch.Tensor,
    "attention_mask": Optional[torch.Tensor]
})


def build_attention_mask(
        local_attention_chunk_tokens: torch.LongTensor, global_attention_chunk_tokens: torch.LongTensor,
        lower_triangular_matrix: torch.Tensor
) -> torch.Tensor:
    """
    For a sequence, we split it into n chunks, which are arranged in the order they appear in the sequence.
    The first n - 1 chunks have only local attention scope, meaning that tokens within a chunk can only see the tokens to their left within the same chunk (including themselves).
    The n-th chunk, however, has a global attention scope, meaning that tokens within this chunk can see all the tokens to their left (including themselves).

    Given local_attention_chunks and global_attention_chunk, num_tokens = sum(local_attention_chunks) + global_attention_chunk.
    Finally, a torch.Tensor matrix of shape [num_tokens, num_tokens] will be returned, which is a lower triangular matrix.

    :param local_attention_chunk_tokens: local_attention_chunks[i] represents the number of tokens in the ith chunk, which has only local attention scope.
    :param global_attention_chunk_tokens: The number of tokens in the chunk that has global attention scope.
    :param lower_triangular_matrix: A lower triangular matrix of shape [num_tokens, num_tokens].

    For the input `local_attention_chunks = [1, 2], global_attention_chunk = 1`, the attention mask matrix should be:
    ```python
    [
        [1, 0, 0, 0]
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 1],
    ]
    ```
    """

    num_tokens = (local_attention_chunk_tokens.sum() + global_attention_chunk_tokens).item()
    attention_mask = torch.zeros(size=(num_tokens, num_tokens), dtype=torch.bool)

    offset = 0
    for n_tokens in local_attention_chunk_tokens:
        attention_mask[offset: offset + n_tokens, offset: offset + n_tokens] = \
            lower_triangular_matrix[:n_tokens, :n_tokens]
        offset += n_tokens
    n_tokens = global_attention_chunk_tokens
    attention_mask[-n_tokens:] = lower_triangular_matrix[num_tokens - n_tokens: num_tokens, :num_tokens]
    return attention_mask


def convert_attention_mask_to_model_required(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    :param attention_mask: torch.bool, [num_tokens, num_tokens]
    :return: torch.bfloat16, [num_tokens, num_tokens]
    """
    attention_mask = attention_mask.to(dtype=torch.bfloat16)
    attention_mask = 1.0 - attention_mask
    attention_mask = attention_mask.masked_fill_(
        mask=attention_mask.to(dtype=torch.bool), value=torch.finfo(torch.bfloat16).min
    )
    return attention_mask


class BlockBatchCollector:
    def __init__(self, pad_token_id: int, max_length: int):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.counter = 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Here we set the batch_size to always be 1.
        return features[0]


class SFTBlockRawDataset:
    def __init__(self, fp: str, model_name: str, max_length: int, tokenizer: PreTrainedTokenizer):
        self.fp = fp
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model_name = model_name

        self.raw_dataset: List[SFTInstance] = []

    def load_dataset(self):
        self.raw_dataset = []
        with open(self.fp, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f), desc=f"Loading dataset {self.fp}"):
                self.raw_dataset.append(json.loads(line))

        if self.max_length != -1:
            self.raw_dataset = [i for i in self.raw_dataset if len(i['inputs']['input_ids']) <= self.max_length]


class SFTBlockDataset(Dataset):
    def __init__(
            self,
            dataset: SFTBlockRawDataset,
            train_prompt: bool,
            train_full_attention: bool,
            add_special_domain_tokens: bool,
            max_length: int
    ):
        super().__init__()
        self.dataset = dataset
        self.raw_dataset = []
        self.num_block_training_cases = 0

        self.train_prompt = train_prompt
        self.train_full_attention = train_full_attention
        self.add_special_domain_tokens: bool = add_special_domain_tokens

        self._block_attention_tokens = self.tokenizer.encode("[Block-Attention]", add_special_tokens=False)
        self._full_attention_tokens = self.tokenizer.encode("[Full-Attention]", add_special_tokens=False)

        self.helper_matrix = torch.tril(
            input=torch.ones(size=[max_length + 64, max_length + 64], dtype=torch.bool),
            diagonal=0
        )

        self._prepare_dataset()

    def _prepare_dataset(self):
        for i, ins in tqdm(
                enumerate(self.dataset.raw_dataset),
                desc=f"Preparing: ",
                total=len(self.dataset.raw_dataset)
        ):
            ins: SFTInstance

            if self.train_full_attention:
                self.raw_dataset.append({
                    "input_ids": np.array([ins["inputs"]["input_ids"]], dtype=np.int64),
                    "labels": np.array([ins["inputs"]["labels"]], dtype=np.int64),
                })

            if not ins["train_block"]:
                continue

            self.num_block_training_cases += 1
            input_ids = ins["block_inputs"]["input_ids"]
            if self.add_special_domain_tokens:
                input_ids = self._block_attention_tokens + input_ids
                ins["block_tokens"][0] += len(self._block_attention_tokens)

            if self.train_prompt:
                labels = input_ids
            else:
                labels = ins["block_inputs"]["labels"]
                if self.add_special_domain_tokens:
                    labels = [-100] * len(self._block_attention_tokens) + labels

            self.raw_dataset.append({
                "input_ids": np.array([input_ids], dtype=np.int64),
                "labels": np.array([labels], dtype=np.int64),
                "block_tokens": np.array(ins["block_tokens"], dtype=np.int64),
                "response_tokens": np.array(ins["response_tokens"], dtype=np.int64)
            })

        random.shuffle(self.raw_dataset)
        print("Block Training Cases: ", self.num_block_training_cases)
        print("Full Attention Training Cases: ", len(self.dataset.raw_dataset))
        print("Block Training Ratio: ", self.num_block_training_cases / len(self.dataset.raw_dataset))
        self.dataset.raw_dataset = []

        gc.collect()
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx % 512 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if "block_tokens" not in self.raw_dataset[idx]:
            return {
                "input_ids": torch.from_numpy(self.raw_dataset[idx]["input_ids"]),
                "labels": torch.from_numpy(self.raw_dataset[idx]["labels"]),
            }

        attention_mask = build_attention_mask(
            local_attention_chunk_tokens=self.raw_dataset[idx]["block_tokens"][:-1],
            global_attention_chunk_tokens=(
                    self.raw_dataset[idx]["block_tokens"][-1] + self.raw_dataset[idx]["response_tokens"]
            ),
            lower_triangular_matrix=self.helper_matrix
        )
        attention_mask = convert_attention_mask_to_model_required(attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(dim=0).unsqueeze(dim=0)
        return {
            "input_ids": torch.from_numpy(self.raw_dataset[idx]["input_ids"]),
            "labels": torch.from_numpy(self.raw_dataset[idx]["labels"]),
            "attention_mask": attention_mask
        }

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self.dataset.tokenizer

    @property
    def max_length(self) -> int:
        return self.dataset.max_length


def get_dataset(
        fp: str,
        model_name: str,
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        train_prompt: bool,
        train_full_attention: bool,
        add_special_domain_tokens: bool
) -> SFTBlockDataset:
    dataset = SFTBlockRawDataset(fp=fp, model_name=model_name, max_length=max_length, tokenizer=tokenizer)
    dataset.load_dataset()
    return SFTBlockDataset(
        dataset=dataset,
        train_prompt=train_prompt,
        train_full_attention=train_full_attention,
        add_special_domain_tokens=add_special_domain_tokens,
        max_length=max_length
    )
