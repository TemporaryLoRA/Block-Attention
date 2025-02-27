import os.path
import sys

sys.path.append(".")

import re
import json
import fire
import random

from tqdm import tqdm
from bson.objectid import ObjectId
from dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Any, List

from define import (
    TuluInstance, MessageWithChunks, TuluInstanceWithChunks, SFTInputs, SFTInstanceWithChunks
)

re_sep_spliter = {
    "\n\n": re.compile(r'(\s*\n\n\s*)'),
    "```": re.compile(r'(```.*?\n.*?```\n+)'),
    "---": re.compile(r'(\s*-{3,}\s*)'),
    "===": re.compile(r'(\s*={3,}\s*)'),
    "\n\t": re.compile(r'(\n\t)'),
    "\n": re.compile(r'(\s*\n\s*)')
}

chunk_prefix_suffix = {
    "system": ("<|system|>\n", "\n"),
    "user": ("<|user|>\n", "\n"),
    "assistant": ("<|assistant|>\n", "<|end_of_text|>\n"),
}


def is_code(text: str) -> bool:
    return any([
        f in text for f in
        [
            "```python", "```cpp", "```c", "```java", "```javascript", "```typescript", "```bash", "```c#", "```sql",
            "```go", "```ruby", "```rust", "```c++", "c++", "c#", "```julia", "```golang"
        ]
    ])


def load_jsonline(fp: str) -> List[Any]:
    with open(fp, 'r', encoding='utf-8') as f:
        return [json.loads(i) for i in f]


def write_jsonline(fp: str, obj: List[Any]):
    with open(fp, 'w', encoding='utf-8') as f:
        for i in obj:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


def load_tulu_dataset(idx: int, data_dir: str = "datahub/tulu3/hf/split") -> List[TuluInstance]:
    fp = os.path.join(data_dir, f"{idx}.jsonline")
    dataset = load_jsonline(fp=fp)
    return [TuluInstance.from_dict(obj=d) for d in dataset]


def _split(x: str, p: re.Pattern) -> List[str]:
    r = p.split(x)
    return [i for i in r if i != ""]


def split_by_delimiter(text: str) -> List[str]:
    if "\n\n" in text:
        results = _split(x=text, p=re_sep_spliter["\n\n"])
        if len(results) == 1:
            return results
        r = [x + y for x, y in zip(results[::2], results[1::2])]
        if len(results) % 2 == 0:
            return r
        return r + [results[-1]]
    if "---" in text:
        return _split(x=text, p=re_sep_spliter["---"])
    if "===" in text:
        return _split(x=text, p=re_sep_spliter["==="])
    if "\n\t" in text:
        return _split(x=text, p=re_sep_spliter["\n\t"])
    if "\n" in text:
        results = _split(x=text, p=re_sep_spliter["\n"])
        if len(results) == 1:
            return results
        r = [x + y for x, y in zip(results[::2], results[1::2])]
        if len(results) % 2 == 0:
            return r
        return r + [results[-1]]
    return [text]


def to_chunks(messages: List[MessageWithChunks]) -> List[str]:
    chunks = []
    for i in range(0, len(messages)):
        prefix, suffix = chunk_prefix_suffix[messages[i].role]
        if len(messages[i].chunks) == 1:
            chunks.append(prefix + messages[i].chunks[0] + suffix)
        else:
            chunks.append(prefix + messages[i].chunks[0])
            chunks.extend(messages[i].chunks[1:-1])
            chunks.append(messages[i].chunks[-1] + suffix)
    return chunks


def to_train_data(ins: TuluInstanceWithChunks, tokenizer: PreTrainedTokenizer) -> List[SFTInstanceWithChunks]:
    data = []
    for i in range(0, len(ins.messages)):
        if ins.messages[i].role != "assistant":
            continue

        conversation = [m.to_dict() for m in ins.messages[:i]]
        prompt = tokenizer.apply_chat_template(conversation=conversation, add_generation_prompt=True, tokenize=False)
        response = ins.messages[i].content + "<|end_of_text|>"

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        if i > 1:
            # Make the last user message have global attention.
            chunks = to_chunks(messages=ins.messages[:i - 1])
            prefix, suffix = chunk_prefix_suffix[ins.messages[i - 1].role]
            chunks.append(prefix + ins.messages[i - 1].content + suffix)
            train_block = True
        else:
            # Split the last user message, the last chunk of the user message has global attention.
            chunks = to_chunks(messages=ins.messages[:i])
            train_block = len(chunks) > 1 and random.random() < 0.45
        if not train_block:
            continue

        chunks[-1] += "<|assistant|>\n"

        block_ids, block_tokens = [], []
        for c in chunks:
            _ids = tokenizer.encode(c, add_special_tokens=False)
            block_ids.extend(_ids)
            block_tokens.append(len(_ids))

        block_input_ids = block_ids + response_ids
        block_labels = [-100] * len(block_ids) + response_ids
        block_inputs = SFTInputs(input_ids=block_input_ids, labels=block_labels)
        data.append(SFTInstanceWithChunks(
            uuid=ObjectId(),
            tulu_uuid=ins.uuid,
            prompt=prompt,
            response=response,
            chunks=chunks,
            inputs=SFTInputs(input_ids=input_ids, labels=labels),
            block_inputs=block_inputs,
            block_tokens=block_tokens,
            response_tokens=len(response_ids),
            # about 23% of tulu3 dataset
            train_block=train_block and not is_code(text=prompt + response)
        ))
    return data


def split_message(m: MessageWithChunks) -> MessageWithChunks:
    m.chunks = split_by_delimiter(m.content)
    return m


def post_process_messages(messages: List[MessageWithChunks]) -> List[MessageWithChunks]:
    if len(messages) >= 12:
        for i in range(0, len(messages) - 1):
            messages[i].chunks = [messages[i].content]
        return messages

    num_chunk_limits = 15
    num_blocks = sum([len(m.chunks) for m in messages])
    while num_blocks >= num_chunk_limits:
        m_i, c_i, c_j, num_chars = -1, -1, -1, 1e8
        for m_idx, m in enumerate(messages):
            if len(m.chunks) < 2:
                continue
            _c_idx_tokens = [(_c_idx, len(c)) for _c_idx, c in enumerate(m.chunks)]
            _c_idx_tokens.sort(key=lambda x: x[1])

            if _c_idx_tokens[0][0] == 0:
                a, b = 0, 1
            else:
                a, b = _c_idx_tokens[0][0] - 1, _c_idx_tokens[0][0]
            if len(m.chunks[a]) + len(m.chunks[b]) < num_chars:
                m_i, c_i, c_j = m_idx, a, b
        if m_i == -1:
            break

        messages[m_i].chunks[c_i] += messages[m_i].chunks[c_j]
        messages[m_i].chunks.pop(c_j)
        num_blocks = sum([len(m.chunks) for m in messages])
    return messages


def process_messages(messages: List[MessageWithChunks]) -> List[MessageWithChunks]:
    messages = [split_message(m=m) for m in messages]
    messages = post_process_messages(messages)
    return messages


def process_tulu_instance(ins: TuluInstance) -> TuluInstanceWithChunks:
    messages = [MessageWithChunks(role=m.role, content=m.content, chunks=[]) for m in ins.messages]
    messages = process_messages(messages=messages)
    return TuluInstanceWithChunks(id=ins.id, messages=messages, source=ins.source, uuid=ins.uuid)


@dataclass
class Args:
    idx: int
    hf_data_dir: str = "datahub/tulu3/hf/"


def main(args: Args):
    random.seed(42)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path='meta-llama/Llama-3.1-8B',
    )
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}"

    dataset: List[TuluInstance] = load_tulu_dataset(idx=args.idx, data_dir=os.path.join(args.hf_data_dir, "split"))
    results: List[TuluInstanceWithChunks] = []
    for i in tqdm(range(0, len(dataset)), total=len(dataset), desc="Process: "):
        ins = process_tulu_instance(ins=dataset[i])
        results.append(ins)

    sft_instances: List[SFTInstanceWithChunks] = []
    for i in tqdm(range(0, len(results)), total=len(results), desc="Process: "):
        sft_instances.extend(to_train_data(ins=results[i], tokenizer=tokenizer))

    sft_instances = [i for i in sft_instances if len(i.inputs.input_ids) < 4096]
    print("len dataset: ", len(sft_instances))

    output_dir = os.path.join(args.hf_data_dir, "block")
    os.system(f"mkdir -p {output_dir}")
    fp = os.path.join(output_dir, f"{args.idx}.jsonline")
    write_jsonline(obj=[r.to_dict() for r in sft_instances], fp=fp)


if __name__ == '__main__':
    args: Args = fire.Fire(component=Args)
    main(args=args)
