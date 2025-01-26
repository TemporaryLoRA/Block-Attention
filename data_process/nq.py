import re
import os
import json
import argparse

import torch
import random

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, TypedDict

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoModel

Document = TypedDict("Document", {"title": str, "text": str, "score": float})

SFTDataInstanceInputs = TypedDict("SFTDataInstanceInputs", {
    "input_ids": List[int],
    "labels": List[int]
})

SFTDataInstance = TypedDict("SFTDataInstance", {
    "prompt": str,
    "question": str,
    "answers": List[str],
    "generated": str,
    "inputs": SFTDataInstanceInputs,
    "documents": List[Document]
})


@dataclass
class BuildArgs:
    eval_fp: str
    output_dir: str


def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = token_embeddings.masked_fill_(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


@torch.no_grad()
def compute_embeddings(sentences: List[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].to(device=model.device, dtype=torch.int64)
    outputs = model(**inputs)
    embeddings = mean_pooling(token_embeddings=outputs[0], mask=inputs["attention_mask"])
    return embeddings


def process_instance(ins: Dict[str, Any]) -> SFTDataInstance:
    documents = [Document(title=i['title'], text=i['text'], score=0.0) for i in ins['ctxs']]
    embeddings = compute_embeddings(
        sentences=[ins['question']] + [i['text'] for i in documents], model=model, tokenizer=retrieval_tokenizer
    )
    q_emb = embeddings[0].clone().unsqueeze(dim=0)
    scores = torch.matmul(input=q_emb, other=embeddings[1:].T).squeeze(dim=0)
    values, indices = torch.sort(input=scores, descending=True)
    values, indices = values.tolist(), indices.tolist()

    for idx, score in zip(indices, values):
        documents[idx]['score'] = score
    documents.sort(key=lambda i: i['score'], reverse=True)

    return SFTDataInstance(
        prompt="",
        question=ins['question'],
        answers=[ins['answer']],
        generated='',
        inputs=SFTDataInstanceInputs(input_ids=[], labels=[]),
        documents=documents[:10]
    )


def tokenizer_instance(ins: SFTDataInstance) -> SFTDataInstance:
    system_prompt = "You are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference documents that may help you in answering the user's question.\n\n"
    for d_idx in range(0, len(ins['documents'])):
        doc = ins["documents"][d_idx]
        system_prompt += f"- Title: {doc['title']}\n{doc['text'].strip()}\n"
    system_prompt = system_prompt.strip()

    user_prompt = f"Please write a high-quality answer for the given question using only the provided search documents (some of which might be irrelevant).\nQuestion: {ins['question']}".strip()
    prompt = llama3_tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    ins["prompt"] = prompt.replace(
        "<|eot_id|><|start_header_id|>user<|end_header_id|>",
        "\n<|eot_id|><|start_header_id|>user<|end_header_id|>"
    )
    return ins


def process_file(input_file: str, output_file: str, num_samples: int):
    with open(input_file, "r", encoding="utf-8") as f:
        nq_instances: List[Dict[str, Any]] = json.load(f)
    if num_samples != -1:
        nq_instances = random.sample(population=nq_instances, k=num_samples)

    dataset: List[SFTDataInstance] = []
    for i in tqdm(range(0, len(nq_instances)), desc="Process NQ: ", total=len(nq_instances)):
        ins = process_instance(ins=nq_instances[i])
        ins = tokenizer_instance(ins=ins)
        dataset.append(ins)

    with open(output_file, "w", encoding="utf-8") as f:
        for i in dataset:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


def parse_args() -> BuildArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_fp", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    return BuildArgs(eval_fp=args.eval_fp, output_dir=args.output_dir)


if __name__ == '__main__':
    args = parse_args()
    os.system(f"mkdir -p {os.path.join(args.output_dir, 'nq_eval')}")

    random.seed(42)
    model_name = "facebook/contriever-msmarco"
    retrieval_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    model: PreTrainedModel = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )

    llama3_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Meta-Llama-3-8B",
        use_fast=False
    )
    if llama3_tokenizer.chat_template is None:
        llama3_tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    process_file(
        input_file=args.eval_fp, output_file=os.path.join(args.output_dir, "nq_eval", "dataset"), num_samples=-1
    )
