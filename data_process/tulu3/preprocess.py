import os
import sys

sys.path.append(".")
import json

import fire
import random
from tqdm import tqdm

from bson.objectid import ObjectId
from transformers import PreTrainedTokenizer, AutoTokenizer

from dataclasses import dataclass
from typing import Any, List

from data_process.tulu3.define import TuluInstance, SFTInstance, SFTInputs


def load_jsonline(fp: str) -> List[Any]:
    with open(fp, 'r', encoding='utf-8') as f:
        return [json.loads(i) for i in f]


def write_jsonline(fp: str, obj: List[Any]):
    with open(fp, 'w', encoding='utf-8') as f:
        for i in obj:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


def load_tulu_dataset_by_idx(idx: int, data_dir: str = "datahub/tulu3/hf/split") -> List[TuluInstance]:
    fp = os.path.join(data_dir, f"{idx}.jsonline")
    dataset = load_jsonline(fp=fp)
    return [TuluInstance.from_dict(obj=i) for i in dataset]


def to_train_data(ins: TuluInstance, tokenizer: PreTrainedTokenizer) -> List[SFTInstance]:
    data = []
    for i in range(0, len(ins.messages)):
        if ins.messages[i].role == "assistant":
            prompt = tokenizer.apply_chat_template(
                conversation=[m.to_dict() for m in ins.messages[:i]],
                add_generation_prompt=True,
                tokenize=False
            )
            response = ins.messages[i].content + "<|end_of_text|>"
            p_input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            r_input_ids = tokenizer.encode(response, add_special_tokens=False)
            input_ids = p_input_ids + r_input_ids
            labels = [-100] * len(p_input_ids) + r_input_ids
            data.append(SFTInstance(
                uuid=ObjectId(),
                tulu_uuid=ins.uuid,
                prompt=prompt,
                response=response,
                inputs=SFTInputs(input_ids=input_ids, labels=labels),
            ))
    return data


@dataclass
class Args:
    idx: int
    hf_data_dir: str = "datahub/tulu3/hf/"


def main(args: Args):
    random.seed(42)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path='meta-llama/Llama-3.1-8B',
    )
    # chat template copy from https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}"
    dataset: List[TuluInstance] = load_tulu_dataset_by_idx(
        idx=args.idx, data_dir=os.path.join(args.hf_data_dir, "split")
    )

    train = []
    for i in tqdm(range(0, len(dataset)), total=len(dataset), desc='To Replicate Data: '):
        r = to_train_data(ins=dataset[i], tokenizer=tokenizer)
        train.extend(r)
    train = [i for i in train if len(i.inputs.input_ids) < 4096]
    print("len tulu3 dataset: ", len(train))

    save_dir = os.path.join(args.hf_data_dir, "sft")
    os.system(f"mkdir -p {save_dir}")
    fp = os.path.join(save_dir, f"{args.idx}.jsonline")
    write_jsonline(fp=fp, obj=[d.to_dict() for d in train])


if __name__ == '__main__':
    args: Args = fire.Fire(component=Args)
    main(args=args)
