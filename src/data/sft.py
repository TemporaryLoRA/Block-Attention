import json
from typing import Dict, List, TypedDict

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

SFTInputs = TypedDict("SFTInputs", {"input_ids": List[int], "labels": List[int]})

SFTInstance = TypedDict("SFTInstance", {
    "prompt": str,
    "response": str,
    "inputs": SFTInputs,
})


class SFTRawDataset:
    def __init__(self, fp: str, max_length: int, tokenizer: PreTrainedTokenizer):
        self.fp = fp
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.raw_dataset: List[SFTInstance] = []

    def load_dataset(self):
        self.raw_dataset = []
        with open(self.fp, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Loading dataset {self.fp}"):
                self.raw_dataset.append(json.loads(line))
        if self.max_length != -1:
            self.raw_dataset = [i for i in self.raw_dataset if len(i['inputs']['input_ids']) <= self.max_length]


class SFTDataset(Dataset):
    def __init__(self, dataset: SFTRawDataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ins = self.dataset.raw_dataset[idx]
        return {
            "input_ids": torch.tensor(data=ins["inputs"]["input_ids"], dtype=torch.int64),
            "labels": torch.tensor(data=ins["inputs"]["labels"], dtype=torch.int64)
        }


def get_dataset(fp: str, max_length: int, tokenizer: PreTrainedTokenizer) -> SFTDataset:
    dataset = SFTRawDataset(fp, max_length, tokenizer)
    dataset.load_dataset()
    return SFTDataset(dataset)
