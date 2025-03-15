import json
import time
from dataclasses import dataclass
from typing import Any
from typing import TypedDict, List, Dict, Tuple, Union

import fire
import requests
from tqdm import tqdm


class Others(TypedDict):
    index: str


class Judge(TypedDict):
    capability: str
    question: str


class Doc(TypedDict):
    question: str
    capability: str
    others: Others
    judge: Judge


class RequestType(TypedDict):
    context: str
    stop_sequences: List[str]
    generation_kwargs: Dict[str, Any]
    generated: str


class RequestItem(TypedDict):
    request_type: str
    request: RequestType


class Instance(TypedDict):
    task_name: str
    label: str
    native_id: int
    doc_id: int
    doc: Doc
    request: List[RequestItem]


def load_json(fp: str) -> List[Any]:
    with open(fp, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(fp: str, obj: List[Any]):
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=4))


def make_block_icl_bbh(prompt: str) -> Tuple[List[str], str]:
    assert "\n\nQ: " in prompt
    pos = prompt.find("\n\nQ: ")
    blocks = []
    while pos != -1:
        blocks.append(prompt[:pos + 2])
        prompt = prompt[pos + 2:]
        pos = prompt.find("\n\nQ: ")
    return blocks, prompt


def make_block_icl_drop(prompt: str) -> Tuple[List[str], str]:
    assert "\n---" in prompt
    pos = prompt.find("\n---")
    blocks = []
    while pos != -1:
        blocks.append(prompt[:pos + 1])
        prompt = prompt[pos + 1:]
        pos = prompt.find("\n---")
    assert blocks[-1].endswith("# Your Task\n\n")
    blocks = blocks[-1][:-len("# Your Task\n\n")]
    prompt = "# Your Task\n\n" + prompt
    return blocks, prompt


def make_block_icl_gsm8k_and_math(prompt: str) -> Tuple[List[str], str]:
    assert "<|end_of_text|>\n" in prompt
    pos = prompt.find("<|end_of_text|>\n")
    blocks = []
    while pos != -1:
        blocks.append(prompt[:pos + len("<|end_of_text|>\n")])
        prompt = prompt[pos + len("<|end_of_text|>\n"):]
        pos = prompt.find("<|end_of_text|>\n")
    return blocks, prompt


def get_response(prompt: str, url: str, task: str) -> str:
    if task in {"bbh", "gsm8k", "math", "drop"}:
        # task type: icl
        if task == "bbh":
            blocks, instruction = make_block_icl_bbh(prompt=prompt)
        elif task == "drop":
            blocks, instruction = make_block_icl_drop(prompt=prompt)
        else:
            blocks, instruction = make_block_icl_gsm8k_and_math(prompt=prompt)
        blocks.append(instruction)
        request_data = {"blocks": blocks}
    else:
        # task type: general
        request_data = {"prompt": prompt, "stream": False}

    r = requests.post(
        url=url,
        data=json.dumps(request_data),
        headers={"Content-Type": "application/json"},
        timeout=600
    )
    return r.json()["generated"]


@dataclass
class Args:
    fp: str
    task: str
    output: str
    server_url: str


def process(args: Args):
    instances: List[Instance] = load_json(fp=args.fp)
    results = []
    for i in tqdm(range(0, len(instances)), total=len(instances), desc=f"Eval {args.task}: "):
        ins = instances[i]
        for j in range(0, len(ins["request"])):
            prompt = ins["request"][j]["request"]["context"]
            generated = get_response(prompt=prompt, url=args.server_url, task=args.task)
            ins["request"][j]["request"]["generated"] = generated
        results.append(ins)
    write_json(fp=args.output, obj=results)


if __name__ == '__main__':
    args: Args = fire.Fire(component=Args)
    process(args=args)
