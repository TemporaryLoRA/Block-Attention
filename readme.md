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

## Processed Data

We have completed the data processing procedure, and below is the processed data. You can directly use it to train the model.


- **Tulu3-Block-FT**/**Tulu3-Rag** Train Data: [download](https://drive.google.com/file/d/17kldAR2CIQPiNJ6ASW9et_GN_wqjFqfv/view?usp=sharing). You can use it to train the two models, Tulu3-Block-FT and Tulu3-RAG.
- **Tulu3-SFT** Train Data: [download](https://drive.google.com/file/d/1hqKcQ3Qbc88WNVlxCfc-illfChc2Hzty/view?usp=sharing). You can use it to train the Tulu3-FT model.

Each line is a training example, and the data format is as follows:

```python
class SFTInputs(TypedDict):
   labels: List[int]
   input_ids: List[int]


class SFTInstanceWithChunks(TypedDict):
   prompt: str
   response: str
   # for sft training 
   inputs: SFTInputs
   # The following fields exist only in the Tulu3-Block-FT Train Data and are used for Block-Attention training.
   chunks: List[str]
   # for block attention training
   block_inputs: SFTInputs
   # the last block has global attention
   block_tokens: List[int]
   response_tokens: int
   # Whether it can be used for block-attention training
   train_block: bool
```

## DataProcess

You can also follow the process below to reprocess the data. Our data is primarily divided into two main parts: **Tulu3** and **RAG**. Below is the processing workflow.

### Tulu3 Dataset

#### Data Downloading

|        Dataset        |                           Source                           |
|:---------------------:|:----------------------------------------------------------:| 
| tulu3-sft-sft-mixture | https://huggingface.co/datasets/allenai/tulu-3-sft-mixture |

#### Data Prepare

**Note**: running following commands to prepare the dataset.

```bash 
mkdir -p datahub/
cd datahub

git lfs install 
git clone https://huggingface.co/datasets/allenai/tulu-3-sft-mixture

mv tulu-3-sft-mixture tulu3
mv tulu3/data tulu3/hf

python3 data_process/tulu3/step0_split.py 
python3 data_process/tulu3/step1_run.py 
python3 data_process/tulu3/step2_run_block.py 
```

After completing the above steps, we will obtain `datahub/tulu3/sft.train` for supervised fine-tuning (SFT) and
`datahub/tulu3/block.train` for our Block-Attention fine-tuning.

### RAG Dataset

#### Data Downloading

|    Dataset    |                                                                   Source                                                                   |
|:-------------:|:------------------------------------------------------------------------------------------------------------------------------------------:|
| 2WikiMultiHop |                                           https://huggingface.co/datasets/xanhho/2WikiMultihopQA                                           |
|    NQ, TQA    | https://github.com/facebookresearch/FiD/blob/main/get-data.sh; https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py |
|      HQA      |                                         https://github.com/hotpotqa/hotpot/blob/master/download.sh                                         |

#### Data Prepare

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
cd datahub
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

#### Data Pre-Processing

1. There is no need to pre-process `2WikiMultiHop` and `HQA. `hotpot_dev_distractor_v1.json` of `HQA` is used for
   following pre-processing steps.
2. After executing get-data.sh, NQ and TQA will call the preprocess.py file from the FiD repository to generate
   processed data files.
3. The `DPR` repository provides Golden Documents (i.e., paragraph snippets that can answer questions) for the `NQ`
   dataset, which are not actually used and can be ignored.

#### Construct Train and Test Set

1. Download retrieval model: [facebook/contriever-msmacro](https://huggingface.co/facebook/contriever-msmarco)
2. Execute following commands for pre-processing

```bash 
mkdir -p datahub/rag

python3 data_process/rag/hqa.py --eval_fp datahub/hqa/hotpot_dev_distractor_v1.json --output_dir datahub/rag

python3 data_process/rag/nq.py --eval_fp datahub/nq/test.json --output_dir datahub/rag

python3 data_process/rag/tqa.py --eval_fp datahub/tqa/test.json --train_fp datahub/tqa/train.json --output_dir datahub/rag

python3 data_process/rag/2wiki.py --dev_fp datahub/2wiki/dev.parquet --train_fp datahub/2wiki/train.parquet --output_dir datahub/rag
```

3. Construct Train Set

After completing the test data projection for each dataset in step 1, additional processing is still required.

1. Execute `data_process/rag/step0_random_sample.py` to randomly sample 20,000 data points from the training data of
   `2wiki`
   and `tqa` to create their respective training sets.

```bash
python3 data_process/rag/step0_random_sample.py --input datahub/rag/2wiki_train/dataset --output datahub/rag/2wiki_train/dataset_p20k --num_samples 20000
python3 data_process/rag/step0_random_sample.py --input cache/tqa_train/dataset --output datahub/rag/tqa_train/dataset_p20k --num_samples 20000
```

2. Execute `data_process/rag/step1_merge.py` to combine the two training files obtained in step 1, resulting in the
   final
   training dataset.

```bash 
python3 data_process/rag/step1_merge.py --inputs "datahub/rag/2wiki_train/dataset_p20k datahub.rag/tqa_train/dataset_p20k" --output datahub/rag/tqa_2wiki_p20k
```

3. Use [Llama3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) to complete the `generated`
   field for training purposes.

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

4. Convert the data structure of RAG into the data structure of Tulu3 to facilitate training.

```bash 
python3 data_process/rag/step2_to_tulu_format.py --input datahub/rag/tqa_2wiki_p20k --output datahub/rag/rag.train
```

5. Mix tulu3 data and rag data

Mix the data of tulu3 and rag to obtain the final block-attention training data.

```bash 
python3 data_process/stepx_merge_tulu3_rag.py --tulu3_input datahub/tulu3/block.train --rag_input datahub/rag/rag.train --output datahub/mix_tulu3_rag.train
```

## Running

Under the `train_scripts/` folder, we have provided three training scripts that can be used to train the following models:

1. **Tulu3-SFT**: `train_scripts/tulu3_sft.sh`, which serves as the base model for the two models below.
2. **Tulu3-RAG**: `train_scripts/tulu3_rag.sh`
3. **Tulu3-Block-FT**: `train_scripts/tulu3_block_ft.sh`

And you need to define the following environment variables in the file:

- `PROJECT_DIR`: Absolute path of the `Block-Attention` project
- `TRAIN_FP`: Training data file, for example `datahub/rag/block.train`
- `EVAL_FP`: Test data file, with the same format as `TRAIN_FP`
- `SAVE_DIR`: Model saving path

## Inference

1. Use `block_generate.py` to obtain the generated results according to the `Block-Attention` method.

```bash
python3 block_generate.py --model_name <the path of block model> --input_file <a jsonline file and each line of JSON has "prompt" field, such as "cache/hqa_eval/dataset">
```

