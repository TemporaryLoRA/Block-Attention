<div align="center">

# Block-Attention for Efficient Prefilling

</div>

<h5 align=center>

[![arXiv](https://img.shields.io/badge/arXiv-2411.16365v3-b31b1b.svg)](https://arxiv.org/abs/2409.15355)
[![hf](https://img.shields.io/badge/🤗-Hugging%20Face-blue.svg)](https://huggingface.co/ldsjmdy)
[![GitHub stars](https://img.shields.io/github/stars/maziao/M2RAG.svg?colorA=orange&colorB=orange&logo=github)](https://github.com/TemporaryLoRA/Block-Attention)

</h5>

Implementation of paper [Block-Attention for Efficient Prefilling](https://arxiv.org/abs/2409.15355).

We introduce Block-attention, an attention mechanism designed to address the increased inference latency and cost in Retrieval-Augmented Generation (RAG) scenarios. 
Traditional approaches often encode the entire context in an auto-regressive manner.
Instead, Block-attention divides retrieved documents into discrete blocks, with each block independently calculating key-value (KV) states except for the final block.
In RAG scenarios, by defining each passage as a block, Block-attention enables us to reuse the KV states of passages that have been seen before, thereby significantly reducing the latency and the computation overhead during inference.
The implementation of Block-attention involves block segmentation, position re-encoding, and fine-tuning the LLM to adapt to the Block-attention mechanism. 
Experiments on 11 diverse benchmarks, including RAG, ICL, and general domains, demonstrate that after block fine-tuning, the Block-attention model not only achieves performance comparable to that of full-attention models, but can also seamlessly switch between the block and full attention modes without any performance loss.
Notably, Block-attention significantly reduces the time to first token (TTFT) and floating point operations (FLOPs) to a very low level. It only takes 45 ms to output the first token for an input sequence with a total length of 32K. Compared to the full-attention models, the TTFT and corresponding FLOPs are reduced by 98.7\% and 99.8\%, respectively. 

Additionally, we also elaborate on how Block-attention is applied in Game AI scenario and the substantial potential benefits it entails. We strongly suggest researchers in the gaming field not to overlook our work.

![Illustration of Block-Attention](./figs/overview.png)

## 🔥 News

- **2025 March 1:** Code, datasets and model weights are released.
- **2024 Sep 15:** Paper available on [arXiv](https://arxiv.org/abs/2409.15355).

## 📋 Table of Contents

- [Block-Attention For Efficient Prefilling](#block-attention-for-efficient-prefilling)
  - [🔥 News](#-news)
  - [📋 Table of Contents](#-table-of-contents)
  - [🤗 Resources](#-resources)
  - [🚀 Getting Started](#-getting-started)
    - [🔧 Data Process](#-data-process)
    - [⚙️ Fine-tuning](#️-fine-tuning-models)
    - [♻️ Inference](#️-inference)
    - [📈 Evaluation](#-evaluation)
  - [📎 Citation](#-citation)

## 🤗 Resources

| Item                                    | Repository                                                                                     |
| --------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Tulu3-sft-mixture Dataset               | [🤗 allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)    |
| Tulu3-SFT Train Dataseet                | [🤗 allenai/tulu-3-sft-mixture](https://drive.google.com/file/d/1hqKcQ3Qbc88WNVlxCfc-illfChc2Hzty/view?usp=sharing)|
| Tulu3-Block-FT / Tulu3-Block-Rag Train Dataset | [💾 Google Drive](https://drive.google.com/file/d/17kldAR2CIQPiNJ6ASW9et_GN_wqjFqfv/view?usp=sharing) |
| Tulu3-Block-FT Model | [🤗 ldsjmdy/Tulu3-Block-FT](https://huggingface.co/ldsjmdy/Tulu3-Block-FT) |
| Tulu3-SFT Model (Baseline) | [🤗 ldsjmdy/Tulu3-SFT](https://huggingface.co/ldsjmdy/Tulu3-SFT) |
| Tulu3-RAG Model (Baseline) | [🤗 ldsjmdy/Tulu3-RAG](https://huggingface.co/ldsjmdy/Tulu3-RAG) |


## 🚀 Getting Started

Although we provided the processed dataset in [🤗 Resources](#-resources), we still release our scripts for processing the raw data as below

---

### 🔧 Data Process

#### 1. Tulu-3 Dataset Process
**Note**: running following commands to prepare the Tulu-3 training dataset.

```bash 
mkdir -p datahub/
cd datahub

git lfs install 
git clone https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
# if you cannot access the dataset through huggingface in mainland china, you could try following commands
# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --repo-type dataset --resume-download allenai/tulu-3-sft-mixture --local-dir allenai/tulu-3-sft-mixture

mv tulu-3-sft-mixture tulu3
mv datahub/tulu3/data datahub/tulu3/hf

python3 data_process/tulu3/step0_split.py 
python3 data_process/tulu3/step1_run.py 
python3 data_process/tulu3/step2_run_block.py 
```

After completing the above steps, we will obtain `datahub/tulu3/sft.train` for supervised fine-tuning (SFT) and
`datahub/tulu3/block.train` for our Block-Attention fine-tuning.

#### 2. RAG Dataset Process

##### 2.1 Data Downloading

|    Dataset    |                                                                   Source                                                                   |
|:-------------:|:------------------------------------------------------------------------------------------------------------------------------------------:|
| 2WikiMultiHop |                                           https://huggingface.co/datasets/xanhho/2WikiMultihopQA                                           |
|    NQ, TQA    | https://github.com/facebookresearch/FiD/blob/main/get-data.sh; https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py |
|      HQA      |                                         https://github.com/hotpotqa/hotpot/blob/master/download.sh                                         |

##### 2.2 Data Prepare

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
# optional: downloading from hf-mirror.com
# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --repo-type dataset --resume-download xanhho/2WikiMultihopQA --local-dir 2WikiMultihopQA
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

##### 2.3 Data Pre-Processing

1. There is no need to pre-process `2WikiMultiHop` and `HQA`. `hotpot_dev_distractor_v1.json` of `HQA` is used for
   following pre-processing steps.
2. After executing get-data.sh, NQ and TQA will call the preprocess.py file from the FiD repository to generate
   processed data files.
3. The `DPR` repository provides Golden Documents (i.e., paragraph snippets that can answer questions) for the `NQ`
   dataset, which are not actually used and can be ignored.

##### 2.4 Construct Train and Test Set

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
      We have already prepared the processed data as below:
         - Tulu3-Block-FT/Tulu3-Block-Rag Train Data: [download](https://drive.google.com/file/d/17kldAR2CIQPiNJ6ASW9et_GN_wqjFqfv/view?usp=sharing). You can use it to train the two models, Tulu3-Block-FT and Tulu3-RAG.
         - Tulu3-SFT Train Data: [download](https://drive.google.com/file/d/1hqKcQ3Qbc88WNVlxCfc-illfChc2Hzty/view?usp=sharing). You can use it to train the Tulu3-FT model.
   
   4. Convert the data structure of RAG into the data structure of Tulu3 to facilitate training.
   
      ```bash 
      python3 data_process/rag/step2_to_tulu_format.py --input datahub/rag/tqa_2wiki_p20k --output datahub/rag/rag.train
      ```
   
   5. Mix tulu3 data and rag data
      
      Mix the data of tulu3 and rag to obtain the final block-attention training data.
      
      ```bash 
      python3 data_process/stepx_merge_tulu3_rag.py --tulu3_input datahub/tulu3/block.train --rag_input datahub/rag/rag.train --output datahub/mix_tulu3_rag.train
      ```

### ⚙️ Fine-tuning Models

Use `train_scripts/block_llama3.sh` to train the `meta-llama/Meta-Llama-3-8B` model in the `Block-Attention` mode.
   And you need to define the following environment variables in the file:

- `PROJECT_DIR`: Absolute path of the `Block-Attention` project
- `TRAIN_FP`: Training data file, for example `datahub/rag/block.train`
- `EVAL_FP`: Test data file, with the same format as `TRAIN_FP`
- `SAVE_DIR`: Model saving path

### ♻️ Inference

Use `block_generate.py` to obtain the generated results according to the `Block-Attention` method.

```bash
python3 block_generate.py --model_name <the path of block model> --input_file <a jsonline file and each line of JSON has "prompt" field, such as "cache/hqa_eval/dataset">
```

### 📈 Evaluation

1. Evaluate RAG results
   Use `eval.py` to conduct the evaluation:
   ```bash
   python3 eval.py --input <the path of file containing evaluation results>
   ```
3. Evaluate general benchmarks
   We leverage [OpenCompass](https://github.com/open-compass/opencompass) to conduct our evaluations on general and RAG benchmarks.

## 📎 Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@misc{sun2024blockattentionefficientrag,
      title={Block-Attention for Efficient RAG}, 
      author={East Sun and Yan Wang and Lan Tian},
      year={2024},
      eprint={2409.15355},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.15355}, 
}
```
