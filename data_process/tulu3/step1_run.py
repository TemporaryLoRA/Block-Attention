import json
import os
import sys
from dataclasses import dataclass
from typing import Any, List

import fire
from tqdm import tqdm

sys.path.append(".")

import multiprocessing as mp


def load_jsonline(fp: str) -> List[Any]:
    with open(fp, 'r', encoding='utf-8') as f:
        return [json.loads(i) for i in f]


def write_jsonline(fp: str, obj: List[Any]):
    with open(fp, 'w', encoding='utf-8') as f:
        for i in obj:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


def run(idx: int, hf_data_dir: str):
    cmd = f"python3 -u data_process/tulu3/preprocess.py --idx {idx} --hf_data_dir {hf_data_dir}"
    os.system(cmd)


@dataclass
class Args:
    hf_data_dir: str = "datahub/tulu3/hf/"
    output: str = "datahub/tulu3/sft.train"
    skip_preprocess: bool = False


def main(args: Args):
    if not args.skip_preprocess:
        handlers = []
        for i in range(0, 1):
            h = mp.Process(target=run, args=(i, args.hf_data_dir))
            h.start()
            handlers.append(h)
        for h in handlers:
            h.join()

    print("Preprocess done, start merge...")

    data_dir = os.path.join(args.hf_data_dir, "sft")
    dataset = []
    for i in tqdm(range(0, 1), desc="Load: ", total=1):
        fp = os.path.join(data_dir, f"{i}.jsonline")
        if not os.path.exists(fp):
            continue
        dataset.extend(load_jsonline(fp=fp))
    write_jsonline(fp=args.output, obj=dataset)


if __name__ == '__main__':
    args = fire.Fire(component=Args)
    main(args=args)
