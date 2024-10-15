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

## Docker 

COMMING SOON ...

## DataProcess

### Data Downloading

|      Dataset      |                                                                     Source                                                                     |
|:-------------:|:------------------------------------------------------------------------------------------------------------------------------------------:|
| 2WikiMultiHop |                                           https://huggingface.co/datasets/xanhho/2WikiMultihopQA                                           |
|    NQ, TQA    | https://github.com/facebookresearch/FiD/blob/main/get-data.sh; https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py |
|      HQA      |                                         https://github.com/hotpotqa/hotpot/blob/master/download.sh                                         |

**Note**: running following commands to prepare the datasets.

First of all, create the save folder for datasets under `Block-Attention` root folder.

```bash
mkdir -p datahub/tqa
#mkdir -p datahub/2wiki
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

# back to Block-Attention
cd Block-Attention/datahub
ln -s FiD/open_domain_data/TQA/test.json tqa/test.json
ln -s FiD/open_domain_data/TQA/train.json tqa/train.json
ln -s FiD/open_domain_data/NQ/test.json nq/test.json
ln -s FiD/open_domain_data/NQ/train.json nq/train.json
```

- HQA

```bash
cd Block-Attention/datahub
mkdir -p hqa
cd hqa 
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

### Data Pre-Processing

1. There is no need to pre-process `2WikiMultiHop` and `HQA. `hotpot_dev_distractor_v1.json` of `HQA` is used for following pre-processing steps.
2. After executing get-data.sh, NQ and TQA will call the preprocess.py file from the FiD repository to generate processed data files.
3. The `DPR` repository provides Golden Documents (i.e., paragraph snippets that can answer questions) for the `NQ` dataset, which are not actually used and can be ignored.

### Construct Train and Test Set

1. Download retrieval model: [facebook/contriever-msmacro](https://huggingface.co/facebook/contriever-msmarco)
2. Execute following commands for pre-processing

    ```bash 
    
    mkdir -p cache
    
    python3 data_process/hqa.py --eval_fp datahub/hqa/hotpot_dev_distractor_v1.json --output_dir cache
    
    python3 data_process/nq.py --eval_fp datahub/nq/test.json --output_dir cache
    
    python3 data_process/tqa.py --eval_fp datahub/tqa/test.json --train_fp datahub/tqa/train.json --output_dir cache
    
    python3 data_process/2wiki.py --dev_fp datahub/2wiki/dev.parquet --train_fp datahub/2wiki/train.parquet --output_dir cache
    ```

3. Construct Train Set

After completing the test data projection for each dataset in step 1, additional processing is still required.

1. Execute `data_process/random_sample.py` to randomly sample 20,000 data points from the training data of `2wiki` and `tqa` to create their respective training sets.

    ```bash
    python3 data_process/random_sample.py --input cache/2wiki_train/dataset --output cache/2wiki_train/dataset_p20k --num_samples 20000
    python3 data_process/random_sample.py --input cache/tqa_train/dataset --output cache/tqa_train/dataset_p20k --num_samples 20000
    ```

2. Execute `data_process/merge.py` to combine the two training files obtained in step 1, resulting in the final training dataset.

    ```bash 
    python3 data_process/merge.py --inputs "cache/2wiki_train/dataset_p20k cache/tqa_train/dataset_p20k" --output cache/tqa_2wiki_p20k
    ```

3. Use ChatGPT to complete the `generated` field for training purposes.
   
    After processing in step 2, the data structure for each line in `tqa_2wiki_p20k` is as follows. Request ChatGPT based on the prompt, and complete the `generated` field with the obtained results.
    Note that some special tokens in the prompt, such as `<|begin_of_text|>`, need to be processed when calling ChatGPT.
    
    ```python
    from typing import TypedDict, List
    
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
    ```

## Running

1. Use `train_scripts/block_llama3.sh` to train the `meta-llama/Meta-Llama-3-8B` model in the `Block-Attention` mode. And you need to define the following environment variables in the file:

- `PROJECT_DIR`: Absolute path of the `Block-Attention` project
- `TRAIN_FP`: Training data file, for example `cache/tqa_2wiki_p20k`
- `EVAL_FP`: Test data file, with the same format as `TRAIN_FP`
- `SAVE_DIR`: Model saving path

## Inference

1. Use `block_generate.py` to obtain the generated results according to the `Block-Attention` method.

```bash
python3 block_generate.py --model_name <the path of block model> --input_file <a jsonline file and each line of JSON has "prompt" field, such as "cache/hqa_eval/dataset">
```

