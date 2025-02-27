import fire
import json
import random
from dataclasses import dataclass


from typing import Any, Dict, List, Union, Optional

@dataclass
class Args:
    inputs: str
    output: str


def load_jsonline(fp: str) -> List[Any]:
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(i) for i in f]


def write_jsonline(fp: str, obj: List[Any]):
    with open(fp, "w", encoding="utf-8") as f:
        for i in obj:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    args: Args = fire.Fire(component=Args)
    random.seed(42)
    ds = []
    for fp in args.inputs.split(" "):
        fp = fp.strip()
        if fp == "":
            continue
        ds.extend(load_jsonline(fp=fp))
    random.shuffle(ds)
    write_jsonline(fp=args.output, obj=ds)