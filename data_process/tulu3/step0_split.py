import json
import os
import random
from dataclasses import dataclass
from typing import Any, List

import fire
import pandas as pd

from define import TuluInstance


def load_tulu_dataset(data_dir: str = "datahub/tulu3/hf"):
    fnames = os.listdir(data_dir)
    dataset = []
    for fname in fnames:
        if not fname.endswith(".parquet"):
            continue
        fp = os.path.join(data_dir, fname)
        r = pd.read_parquet(path=fp)
        dataset.extend(r.to_dict(orient="records"))
    print("len tulu dataset: ", len(dataset))
    return [TuluInstance.from_dict(obj=d) for d in dataset]


def write_jsonline(fp: str, obj: List[Any]):
    with open(fp, 'w', encoding='utf-8') as f:
        for i in obj:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


@dataclass
class Args:
    hf_data_dir: str = "datahub/tulu3/hf"


if __name__ == '__main__':
    args: Args = fire.Fire(component=Args)
    dataset: List[TuluInstance] = load_tulu_dataset(data_dir=args.hf_data_dir)

    random.seed(42)
    random.shuffle(dataset)

    num_splits = 128
    splits: List[List[TuluInstance]] = [[] for _ in range(num_splits)]
    for i in range(0, len(dataset)):
        splits[i % num_splits].append(dataset[i])

    save_dir = os.path.join(args.hf_data_dir, "split")
    os.system(f"mkdir -p {save_dir}")

    for i in range(0, len(splits)):
        write_jsonline(fp=os.path.join(save_dir, f"{i}.jsonline"), obj=[t.to_dict() for t in splits[i]])
    print("write done")
