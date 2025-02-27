import sys

sys.path.append(".")

import re
import json
import fire
import random

from tqdm import tqdm
from bson.objectid import ObjectId
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Any, Set, Dict, List, Tuple, Union, Optional, Literal, NamedTuple, TypedDict, Callable

from data_process.tulu3.define import SFTInstanceWithChunks, SFTInputs


def load_jsonline(fp: str) -> List[Any]:
    with open(fp, 'r', encoding='utf-8') as f:
        return [json.loads(i) for i in f]


def write_jsonline(fp: str, obj: List[Any]):
    with open(fp, 'w', encoding='utf-8') as f:
        for i in obj:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


def replace_prompt_to_tulu_format(prompt: str) -> str:
    prompt = prompt.replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "<|user|>\n")
    prompt = prompt.replace("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "\n\n")
    prompt = prompt.replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "\n<|assistant|>\n")
    return prompt


def make_blocks(prompt: str) -> Tuple[List[str], str]:
    blocks: List[str] = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n"
    ]
    assert prompt.startswith(blocks[0]), json.dumps({
        "prompt": prompt, "blocks": blocks[0]}, ensure_ascii=False, indent=4
    )
    content = prompt[len(blocks[0]):]

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
    instruction_ans_response = "<|eot_id|>" + instruction_ans_response

    assert instruction_ans_response.startswith(
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Please write a high-quality answer for the given question using only the provided search documents"
    )
    blocks = [b for b in blocks if b != ""]

    # 为了和 Tulu 的 chat_template 对齐，进行如下的处理：
    assert "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" in blocks[0], blocks[0]
    blocks[0] = blocks[0].replace(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        "<|user|>\n"
    )
    assert "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" in instruction_ans_response, instruction_ans_response
    instruction_ans_response = instruction_ans_response.replace(
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "\n\n"
    )
    assert "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" in instruction_ans_response, instruction_ans_response
    instruction_ans_response = instruction_ans_response.replace(
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "\n<|assistant|>\n"
    )
    return blocks, instruction_ans_response


def to_train_data(ins: Dict[str, Any]) -> List[SFTInstanceWithChunks]:
    blocks, instruct_and_response = make_blocks(prompt=ins["prompt"])
    blocks.append(instruct_and_response)

    tulu_prompt = replace_prompt_to_tulu_format(prompt=ins["prompt"])
    assert tulu_prompt == "".join(blocks)

    response: str = ins["generated"]
    if response.endswith("<|eot_id|"):
        response = response[:-len("<|eot_id|")] + "<|end_of_text|>"
    else:
        response = response + "<|end_of_text|>"

    prompt_input_ids = tokenizer.encode(tulu_prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    input_ids = prompt_input_ids + response_ids
    labels = [-100] * len(prompt_input_ids) + response_ids
    sft_inputs = SFTInputs(input_ids=input_ids, labels=labels)

    block_ids, block_tokens = [], []
    for b in blocks:
        _ids = tokenizer.encode(b, add_special_tokens=False)
        block_ids.extend(_ids)
        block_tokens.append(len(_ids))
    block_input_ids = block_ids + response_ids
    block_labels = [-100] * len(block_ids) + response_ids
    block_inputs = SFTInputs(input_ids=block_input_ids, labels=block_labels)

    return [
        SFTInstanceWithChunks(
            uuid=ObjectId(),
            tulu_uuid=ObjectId(),
            prompt=tulu_prompt,
            response=response,
            inputs=sft_inputs,
            chunks=blocks,
            block_tokens=block_tokens,
            response_tokens=len(response_ids),
            block_inputs=block_inputs,
            train_block=True
        )
    ]


@dataclass
class Args:
    input: str = "datahub/rag/tqa_2wiki_p20k"
    output: str = "datahub/rag/block.train"


def process(args: Args):
    dataset = load_jsonline(fp=args.input)
    dataset = [to_train_data(ins=i)[0].to_dict() for i in tqdm(dataset, total=len(dataset), desc='Convert: ')]
    write_jsonline(obj=dataset, fp=args.output)


if __name__ == '__main__':
    args: Args = fire.Fire(component=Args)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="meta-llama/LLama-3.1-8B",
        use_fast=False
    )
    process(args=args)
