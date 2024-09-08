# BLOCK-ATTENTION FOR LOW-LATENCY RAG

We introduce Block-Attention, an attention mechanism designed to address the increased inference latency in Retrieval-Augmented Generation (RAG) scenarios.
Its main idea lies in dividing the input sequence into blocks, where each block calculates its key-value (KV) states independently except for the final block. In RAG scenarios, by defining each passage as a block, Block-Attention enables us to pre-compute the KV states for all passages and cache them in memory.
The implementation involves block segmentation, positional encoding calculation, and fine-tuning the LLM to adapt to the Block-Attention mechanism. Experiments on four RAG benchmarks demonstrate that after block fine-tuning, the Block Attention model can achieve performance comparable to (68.4 vs 67.9 on Llama3) or even better (62.8 vs 59.6 on Mistral) than self-attention models. Notably, BlockAttention reduces the TTFT to a very low level. It only takes 45 ms to output the first token for an input sequence with a total length of 32K. Compared with the self-attention model, the time consumption is reduced by 98.7%.

## Code

Code is coming soon……