import json
import random
from dataclasses import dataclass
from typing import Any, List

from tqdm import tqdm


def load_jsonline(fp: str) -> List[Any]:
    with open(fp, 'r', encoding='utf-8') as f:
        return [json.loads(i) for i in tqdm(f)]


def write_jsonline(fp: str, obj: List[Any]):
    with open(fp, 'w', encoding='utf-8') as f:
        for i in obj:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


@dataclass
class Args:
    rag_input: str
    tulu3_input: str
    output: str


def main(args: Args):
    random.seed(42)
    rag_dataset = load_jsonline(fp=args.rag_input)
    tulu3_dataset = load_jsonline(fp=args.tulu3_input)
    # Take an equal amount of tulu3 data as the rag data.
    tulu3_dataset = random.sample(population=tulu3_dataset, k=len(rag_dataset))
    dataset = rag_dataset + tulu3_dataset
    random.shuffle(dataset)
    write_jsonline(fp=args.output, obj=dataset)


if __name__ == '__main__':
    main()
