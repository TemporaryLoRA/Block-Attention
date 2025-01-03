import json
import argparse

import torch
from torch.nn import functional as F

from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig, LlamaForCausalLM

from transformers import (
    AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM, PreTrainedModel, GenerationConfig, set_seed, AutoConfig
)


SFTDataInstanceInputs = TypedDict("SFTDataInstanceInputs", {
    "input_ids": List[int],
    "labels": List[int]
})

SFTDataInstance = TypedDict("SFTDataInstance", {
    "prompt": str,
    "answers": List[str],
    "generated": str,
    "inputs": SFTDataInstanceInputs
})


def pkv_to_device(pkv: DynamicCache, device: Union[torch.device, str]) -> DynamicCache:
    for i in range(0, len(pkv.key_cache)):
        pkv.key_cache[i] = pkv.key_cache[i].to(device=device)
        pkv.value_cache[i] = pkv.value_cache[i].to(device=device)
    return pkv


def rotate_half(x):
    """
    transformers.models.llama.modeling_llama.rotate_half
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat(tensors=(-x2, x1), dim=-1)


def apply_rotary_pos_emb(k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed.to(dtype=torch.bfloat16)

def apply_pkv_rotary_position_embeddings(pkv: DynamicCache, emb: LlamaRotaryEmbedding) -> DynamicCache:
    device = pkv.key_cache[0].device
    emb.to(device=device)
    position_ids = torch.arange(start=0, end=pkv.key_cache[0].size(-2), dtype=torch.int64, device=device)
    position_ids = position_ids.unsqueeze(dim=0).repeat(repeats=[pkv.key_cache[0].size(0), 1])
    cos, sin = emb(x=pkv.key_cache[0].to(dtype=torch.float32), position_ids=position_ids)
    for i in range(0, len(pkv.key_cache)):
        pkv.key_cache[i] = apply_rotary_pos_emb(
            k=pkv.key_cache[i].to(dtype=torch.float32), cos=cos, sin=sin, position_ids=position_ids
        )
    return pkv


def apply_pkv_rerotary_position_embeddings(pkv: DynamicCache, emb: LlamaRotaryEmbedding) -> DynamicCache:
    device = pkv.key_cache[0].device
    emb.to(device=device)
    position_ids = torch.arange(start=0, end=pkv.key_cache[0].size(-2), dtype=torch.int64, device=device)
    position_ids = position_ids.unsqueeze(dim=0).repeat(repeats=[pkv.key_cache[0].size(0), 1])
    cos, sin = emb(x=pkv.key_cache[0].to(dtype=torch.float32), position_ids=position_ids)
    for i in range(0, len(pkv.key_cache)):
        pkv.key_cache[i] = apply_rotary_pos_emb(
            k=pkv.key_cache[i].to(dtype=torch.float32), cos=cos, sin=-sin, position_ids=position_ids
        )
    return pkv


def merge_and_rotary_past_key_values(pkvs: List[DynamicCache], emb: LlamaRotaryEmbedding) -> DynamicCache:
    cache = pkvs[0]
    for l_idx in range(0, len(cache)):
        cache.key_cache[l_idx] = torch.cat(
            tensors=[cache.key_cache[l_idx]] + [pkvs[b_idx].key_cache[l_idx] for b_idx in range(1, len(pkvs))],
            dim=-2
        )
        cache.value_cache[l_idx] = torch.cat(
            tensors=[cache.value_cache[l_idx]] + [pkvs[b_idx].value_cache[l_idx] for b_idx in range(1, len(pkvs))],
            dim=-2
        )
    cache = apply_pkv_rotary_position_embeddings(pkv=cache, emb=emb)
    return cache


@torch.no_grad()
def build_block_past_key_values(
        prompt: str, tokenizer: PreTrainedTokenizer, model: LlamaForCausalLM, emb: LlamaRotaryEmbedding
) -> Tuple[List[DynamicCache], torch.Tensor]:
    blocks: List[str] = [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n"
    ]
    assert prompt.startswith(blocks[0])
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

    caches: List[DynamicCache] = []
    input_ids = None
    for b_idx, block in enumerate(blocks):
        block_input_ids = torch.tensor(
            data=[tokenizer.encode(block, add_special_tokens=False)],
            dtype=torch.int64,
            device=model.device
        )
        if b_idx == 0:
            input_ids = block_input_ids
        else:
            input_ids = torch.cat(tensors=[input_ids, block_input_ids], dim=-1)

        output: CausalLMOutputWithPast = model(
            input_ids=block_input_ids, use_cache=True, past_key_values=DynamicCache(), return_dict=True
        )
        pkv = apply_pkv_rerotary_position_embeddings(pkv=output.past_key_values, emb=emb)
        caches.append(pkv)
    response_input_ids = torch.tensor(
        data=[tokenizer.encode(instruction_ans_response, add_special_tokens=False)],
        dtype=torch.int64,
        device=model.device
    )
    input_ids = torch.cat(tensors=[input_ids, response_input_ids], dim=-1)
    return caches, input_ids


@torch.no_grad()
def block_generate(
        prompt: str, generation_config: GenerationConfig, model: LlamaForCausalLM, emb: LlamaRotaryEmbedding,
        tokenizer: PreTrainedTokenizer
) -> str:
    past_key_values, input_ids = build_block_past_key_values(
        prompt=prompt, tokenizer=tokenizer, model=model, emb=emb
    )
    past_key_values = merge_and_rotary_past_key_values(pkvs=past_key_values, emb=emb)
    input_length = input_ids.size(-1)

    outputs = model.generate(
        input_ids=input_ids, generation_config=generation_config, past_key_values=past_key_values,
        use_cache=True, eos_token_id=[tokenizer.eos_token_id], tokenizer=tokenizer
    )
    return tokenizer.decode(token_ids=outputs[0][input_length:].tolist())


@dataclass
class Args:
    model_name: str
    input_file: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--input_file", type=str)
    args = parser.parse_args()
    return Args(model_name=args.model_name, input_file=args.input_file)


def main():
    args = parse_args()
    set_seed(seed=42)

    with open(args.input_file, "r", encoding='utf-8') as f:
        dataset: List[SFTDataInstance] = [json.loads(i) for i in f ]

    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2"
    )
    config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.model_name)
    emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(
        dim=config.hidden_size // config.num_attention_heads,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta
    ).to(device=model.device, dtype=torch.float32)
    model.eval()
    emb.eval()

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        use_fast=False
    )

    generation_config = GenerationConfig(
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.0,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=200,
        stop_strings=['<|im_end|>', "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"]
    )

    for i in dataset:
        generated = block_generate(
            prompt=i["prompt"], generation_config=generation_config, model=model, emb=emb, tokenizer=tokenizer
        )
        print("Prompt:")
        print(i["prompt"])
        print("Generated: ")
        print(generated)
        input()


if __name__ == '__main__':
    main()
    from transformers.training_args import TrainingArguments