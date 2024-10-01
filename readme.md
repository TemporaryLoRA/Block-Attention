
# BLOCK-ATTENTION FOR EFFICIENT RAG

This is the repository for the [BLOCK-ATTENTION FOR EFFICIENT RAG](https://arxiv.org/abs/2409.15355).

We introduce Block-Attention, an attention mechanism designed to address the increased inference latency and cost in Retrieval-Augmented Generation (RAG) scenarios. Unlike existing works that encodes the whole context, its main idea lies in dividing the retrieved documents into blocks, where each block calculates key-value (KV) states independently except for the final block. In RAG scenarios, by defining each passage as a block, Block-Attention enables us to pre-compute the KV states for all passages and cache them in memory, significantly reducing the latency and the computation cost during inference. The implementation involves block segmentation, positional encoding calculation, and fine-tuning the LLM to adapt to the Block-Attention mechanism. Experiments on four RAG benchmarks demonstrate that after block fine-tuning, the Block Attention model can achieve performance comparable to (68.4\% vs 67.9\% on Llama3) or even better (62.8\% vs 59.6\% on Mistral) than self-attention models. Notably, Block-Attention reduces the TTFT (the time to first token) and FLOPs (floating point operations) to a very low level. It only takes 45 ms to output the first token for an input sequence with a total length of 32K. Compared with the self-attention model, the time consumption and corresponding FLOPs are reduced by 98.7\% and 99.8\%, respectively. 

## Running

1. Use the `data_process/2wiki.py` script to generate the dataset.

```bash 
python3 data_process/2wiki.py --train_fp <the path of 2wiki train dataset> --eval_fp <the path of 2wiki dev dataset> --output_dir <the path of output dir>
```

2. Use `train_scripts/block_llama3.sh` to train the `meta-llama/Meta-Llama-3-8B` model in the `Block-Attention` mode.

3. Use `block_generate.py` to obtain the generated results according to the `Block-Attention` method.

```bash
python3 block_generate.py --model_name <the path of block model> --input_file <a jsonline file and each line of JSON has "prompt" field>
```
