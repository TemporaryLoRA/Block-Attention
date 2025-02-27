import fire
import json
import random
from dataclasses import dataclass


from typing import Any, Dict, List, Union, Optional

@dataclass
class Args:
    input: str
    output: str
    num_samples: int


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
    ds = load_jsonline(fp=args.input)
    ds = random.sample(population=ds, k=args.num_samples)
    write_jsonline(fp=args.output, obj=ds)

