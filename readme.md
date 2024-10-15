# BLOCK-ATTENTION FOR EFFICIENT RAG

This is the repository for the [BLOCK-ATTENTION FOR EFFICIENT RAG](https://arxiv.org/abs/2409.15355).

We introduce Block-Attention, an attention mechanism designed to address the increased inference latency and cost in
Retrieval-Augmented Generation (RAG) scenarios. Unlike existing works that encodes the whole context, its main idea lies
in dividing the retrieved documents into blocks, where each block calculates key-value (KV) states independently except
for the final block. In RAG scenarios, by defining each passage as a block, Block-Attention enables us to pre-compute
the KV states for all passages and cache them in memory, significantly reducing the latency and the computation cost
during inference. The implementation involves block segmentation, positional encoding calculation, and fine-tuning the
LLM to adapt to the Block-Attention mechanism. Experiments on four RAG benchmarks demonstrate that after block
fine-tuning, the Block Attention model can achieve performance comparable to (68.4\% vs 67.9\% on Llama3) or even
better (62.8\% vs 59.6\% on Mistral) than self-attention models. Notably, Block-Attention reduces the TTFT (the time to
first token) and FLOPs (floating point operations) to a very low level. It only takes 45 ms to output the first token
for an input sequence with a total length of 32K. Compared with the self-attention model, the time consumption and
corresponding FLOPs are reduced by 98.7\% and 99.8\%, respectively.

## Docker 构建

1. Dockerfile依赖nvidia镜像`nvidia/cuda:11.8.0-devel-ubuntu22.04`，需要先执行命令：

```bash
docker pull nvidia/cuda:11.8.0-devel-ubuntu22.04
```

2. 在Block-Attention项目文件夹内部执行

```bash
docker build -t block_attention:latest . 
```

## DataProcess

### 数据下载

论文所用数据的原始来源见：

|      数据集      |                                                                     来源                                                                     |
|:-------------:|:------------------------------------------------------------------------------------------------------------------------------------------:|
| 2WikiMultiHop |                                           https://huggingface.co/datasets/xanhho/2WikiMultihopQA                                           |
|    NQ, TQA    | https://github.com/facebookresearch/FiD/blob/main/get-data.sh; https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py |
|      HQA      |                                         https://github.com/hotpotqa/hotpot/blob/master/download.sh                                         |

**注：**可分别执行以下命令下载文件

首先在`Block-Attention`下创建各个数据集对应的存储路径。

```bash
mkdir -p datahub/tqa
mkdir -p datahub/2wiki
mkdir -p datahub/nq
mkdir -p datahub/hqa
```

- 2WikiMultiHop

```bash
cd Block-Attention/datahub
git lfs install 
git clone https://huggingface.co/datasets/xanhho/2WikiMultihopQA
ln -s 2WikiMultihopQA 2wiki
```

- NQ、TQA

```bash 
cd Block-Attention/datahub 
git clone https://github.com/facebookresearch/FiD

cd FiD
bash get-data.sh 

```

- HQA

```bash
cd Block-Attention/datahub
mkdir -p hqa
cd hqa 
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

### 数据预处理

`2WikiMultiHop`和`HQA`不需要处理，`HQA`选择下载下来的`hotpot_dev_distractor_v1.json`文件。

`NQ`和`TQA`在执行`get-data.sh`后，会调用`FiD`仓库的`preprocess.py`文件，生成处理之后的数据文件。

`DPR`仓库提供了`NQ`数据集的Golden Document(即能够回答的问题的段落片段)，没有实际用到，可以忽略不计。

### 训练、测试数据集构建

1. 下载retrieval模型

- facebook/contriever-msmacro：https://huggingface.co/facebook/contriever-msmarco

1. 分别执行以下命令

```bash 

mkdir -p cache

python3 data_process/hqa.py --eval_fp datahub/hqa/hotpot_dev_distractor_v1.json --output_dir cache

python3 data_process/nq.py --eval_fp <> --output_dir cache

python3 data_process/tqa.py --eval_fp <> --train_fp <> --output_dir cache

python3 data_process/2wiki.py --eval_fp datahub/2wiki/dev.parquet --train_fp datahub/2wiki/train.parquet --output_dir cache
```

2. 构建训练集

经过步骤 1 各数据集的测试数据极影完成，还要额外处理一下。

1. 执行`data_process/random_sample.py`，从`2wiki`和`tqa`的训练数据中随机sample 20,000条数据组建各自的训练集
2. 执行`data_process/merge.py`，将步骤 1 得到的两个训练文件合并 ，得到最终的训练数据集


## Running

1. Use `train_scripts/block_llama3.sh` to train the `meta-llama/Meta-Llama-3-8B` model in the `Block-Attention` mode.

2. Use `block_generate.py` to obtain the generated results according to the `Block-Attention` method.

```bash
python3 block_generate.py --model_name <the path of block model> --input_file <a jsonline file and each line of JSON has "prompt" field>
```

