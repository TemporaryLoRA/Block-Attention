import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict

from torch.onnx.symbolic_opset11 import chunk
from transformers import PreTrainedTokenizer


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class MessageWithBlocks:
    role: Literal["system", "user", "assistant"]
    content: str
    blocks: List[str]


SFTInputs = TypedDict("SFTInputs", {
    "input_ids": List[int],
    "labels": List[int]
})

SFTInstance = TypedDict("SFTInstance", {
    "inputs": SFTInputs,
    "blocks": List[str],
    "block_inputs": SFTInputs,
    "block_tokens": List[int],
    "response_tokens": int,
    "train_block": bool
})

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


def process_blocks(ins: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> SFTInstance:
    if "inputs" in ins:
        return ins

    if "chunks" in ins:
        ins["block"] = ins.pop("chunks")

    prompt, response, blocks = ins["prompt"], ins["response"], ins["blocks"]

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    inputs = SFTInputs(input_ids=input_ids, labels=labels)

    block_ids, block_tokens = [], []
    for b in blocks:
        _ids = tokenizer.encode(b, add_special_tokens=False)
        block_ids.extend(_ids)
        block_tokens.append(len(_ids))

    block_input_ids = block_ids + response_ids
    block_labels = [-100] * len(block_ids) + response_ids
    block_inputs = SFTInputs(input_ids=block_input_ids, labels=block_labels)
    return SFTInstance(
        inputs=inputs,
        blocks=blocks,
        block_inputs=block_inputs,
        block_tokens=block_tokens,
        response_tokens=len(response_ids),
        train_block=True
    )


def _split(x: str, p: re.Pattern) -> List[str]:
    r = p.split(x)
    return [i for i in r if i != ""]


def _split_by_delimiter(text: str) -> List[str]:
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


def _merge_message_blocks(messages: List[MessageWithBlocks], num_blocks_limit: int) -> List[MessageWithBlocks]:
    """
    Keep the number of blocks within the num_block_limits.
    """
    if num_blocks_limit == -1:
        return messages

    if len(messages) >= num_blocks_limit:
        for i in range(0, len(messages) - 1):
            messages[i].blocks = [messages[i].content]
        return messages

    num_blocks = sum([len(m.blocks) for m in messages])
    while num_blocks >= num_blocks_limit:
        m_i, b_i, b_j, num_chars = -1, -1, -1, 1e8
        for m_idx, m in enumerate(messages):
            if len(m.blocks) < 2:
                continue
            _b_idx_tokens = [(_c_idx, len(c)) for _c_idx, c in enumerate(m.blocks)]
            _b_idx_tokens.sort(key=lambda x: x[1])

            if _b_idx_tokens[0][0] == 0:
                a, b = 0, 1
            else:
                a, b = _b_idx_tokens[0][0] - 1, _b_idx_tokens[0][0]
            if len(m.blocks[a]) + len(m.blocks[b]) < num_chars:
                m_i, b_i, b_j = m_idx, a, b
        if m_i == -1:
            break

        messages[m_i].blocks[b_i] += messages[m_i].blocks[b_j]
        messages[m_i].blocks.pop(b_j)
        num_blocks = sum([len(m.blocks) for m in messages])
    return messages


def _to_blocks(messages: List[MessageWithBlocks]) -> List[str]:
    blocks = []
    for i in range(0, len(messages)):
        prefix, suffix = chunk_prefix_suffix[messages[i].role]
        if len(messages[i].blocks) == 1:
            blocks.append(prefix + messages[i].blocks[0] + suffix)
        else:
            blocks.append(prefix + messages[i].blocks[0])
            blocks.extend(messages[i].blocks[1:-1])
            blocks.append(messages[i].blocks[-1] + suffix)
    return blocks


def _to_train_data(messages: List[MessageWithBlocks], tokenizer: PreTrainedTokenizer) -> List[SFTInstance]:
    data = []
    for i in range(0, len(messages)):
        if messages[i].role != "assistant":
            continue

        conversation = [{"role": m.role, "content": m.content} for m in messages[:i]]
        prompt = tokenizer.apply_chat_template(conversation=conversation, add_generation_prompt=True, tokenize=False)
        response = messages[i].content + "<|end_of_text|>"

        blocks = _to_blocks(messages=messages[:i])
        blocks[-1] += "<|assistant|>\n"

        ins = process_blocks(ins={"prompt": prompt, "response": response, "blocks": blocks}, tokenizer=tokenizer)
        data.append(ins)
    return data


def process_messages(
        messages: List[Dict[str, Any]], num_blocks_limit: int, tokenizer: PreTrainedTokenizer
) -> List[SFTInstance]:
    messages: List[Message] = [Message(**m) for m in messages]
    messages: List[MessageWithBlocks] = [
        MessageWithBlocks(role=m.role, content=m.content, blocks=_split_by_delimiter(text=m.content)) for m in messages
    ]
    messages: List[MessageWithBlocks] = _merge_message_blocks(messages=messages, num_blocks_limit=num_blocks_limit)
    return _to_train_data(messages=messages, tokenizer=tokenizer)
