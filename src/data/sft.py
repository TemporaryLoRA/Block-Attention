import json
from tqdm import tqdm
from typing import Any, Dict, List, Union, TypedDict

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

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


class SFTRawDataset:
    def __init__(self, fp: str, max_length: int):
        self.fp = fp
        self.raw_dataset: List[SFTDataInstance] = []
        self.max_length = max_length

    def load_dataset(self):
        self.raw_dataset.clear()
        with open(self.fp, "r", encoding="utf-8") as f:
            items: List[SFTDataInstance] = [json.loads(i) for i in f]
        self.raw_dataset = [i for i in items if len(i["inputs"]['input_ids']) < self.max_length]


class SFTDataset(Dataset):
    def __init__(self, dataset: SFTRawDataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ins = self.dataset.raw_dataset[idx]
        return {
            "input_ids": torch.tensor(data=[ins["inputs"]["input_ids"]], dtype=torch.int64),
            "labels": torch.tensor(data=[ins["inputs"]["labels"]], dtype=torch.int64)
        }


def get_dataset(fp: str, max_length: int) -> SFTDataset:
    dataset = SFTRawDataset(fp=fp, max_length=max_length)
    dataset.load_dataset()
    return SFTDataset(dataset=dataset)
