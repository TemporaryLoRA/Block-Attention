# Evaluate on OpenCompass

This is the tutorial for conducting the evaluation on the opencompass, consisting of following steps.

## 1. Prepare Input Data

1. Please prepare the opencompass conda env by following [their tutorial](https://github.com/open-compass/opencompass)
2. Please download the OpenCompass core set:

    ```bash
    # Download dataset to data/ folder
    wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
    unzip OpenCompassData-core-20240207.zip
    ```

3. Please process benchmarks using opencompass evaluation configs

    ```bash
    cd block_attention
    # remember to replace the `data_root_path` and `data_output_path` in the script
    python process_data.py
    ```

   The data process follows the opencompass evaluation config (`block_attention/config.py`). As defined in this config,
   the GSM8K, MATH, BBH and DROP benchmarks are evaluated under In-Context Learning (ICL) settings, while IFEval,
   HumanEval and MMLU are evaluated in zero-shot manner.
   We have already provided the processed dataset in `block_attention/data.tgz`

   In each processed data file, the evaluation examples are in following format (GSM8K example):
    ```json
    [
        {
            "task_name": "gsm8k",
            "label": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18",
            "native_id": 0,
            "doc_id": 0,
            "doc": {
                "question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18"
            },
            "request": [
                {
                    "request_type": "generate_util",
                    "request": {
                        "context": "...... Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nLet's think step by step\nAnswer:\n<|assistant|>\n"
                        "stop_sequences": [
                            "Question",
                            "</s>",
                            "<|im_end|>"
                        ],
                        "generation_kwargs": {
                            "max_gen_toks": 2048,
                            "do_sample": false,
                            "temperature": 0.0
                        }
                    }
                }
            ]
        },
        ...
    ]
    ```
   where `doc` contains the raw data of this example and `doc_id` contains its index.
   `label` saves the ground-truth answer.
   `request` contains the prompt and generation hyper-parameters for LLM's inference.

## 2. Inference LLMs on the Processed Data

1. Start the Inference Service

   For ICL tasks (gsm8k, bbh, drop, math), it performs inference using block attention:

    ```bash
    python3 server/block_generate_server.py --model <the path of your block-ft model> --port 8080 --dtype bfloat16
    ```

   For other tasks, you can use vLLM for accelerated inference:

    ```bash 
    python3 server/vllm_server.py --model <the path of your model> --host <LOCAL IP> --dtype bfloat16 --port 8080 --tokenizer-mode slow --tensor-parallel-size 1
    ```

2. Begin Inference

   ```bash
   cd evaluation_oc;
   
   python3 inference.py --input $INPUT_FILE --output $OUTPUT_FILE --task $TASK --server_url http://$LOCAL_IP:8080/generate
   # INPUT_FILE: Path to the file processed in Step 1
   # OUTPUT_FILE: Path to save the result file, which should be located under the `block_attention/convert_generation_to_oc_format/data` folder, e.g., `block_attention/convert_generation_to_oc_format/data/gsm8k.json`
   # TASK: Task name, e.g., `gsm8k`, 'bbh', 'drop', etc.
   ```

## 3. Evaluate on output dataset

1. Convert generated data to opencompass format

   After collecting the generated output data, we need to convert it to the opencompass evaluation format:

    ```bash
    cd block_attention/convert_generation_to_oc_format;
    # this scripts will read the generated output data (`block_attention/convert_generation_to_oc_format/data`) and write the processed data into `block_attention/convert_generation_to_oc_format/output` 
    python convert.py
    ```

2. Put the converted data under `outputs`

    ```bash
    mkdir -p outputs/default/block_attention/predictions/
    cp -r block_attention/convert_generation_to_oc_format/output/* outputs/default/block_attention/predictions
    ```

3. Evaluate the results with OpenCompass

    ```bash
    ./eval.sh
    ```

   The evaluated models are listed at these two configs:
    * `opencompass/configs/models/hf_internlm/tulu.py`
    * `opencompass/opencompass/configs/models/hf_internlm/tulu.py`

   In opencompass, the evaluated models are detected by the `abbr` key.

    ```python
    from opencompass.models import HuggingFacewithChatTemplate

    models = [
        dict(
            type=HuggingFacewithChatTemplate,
            abbr='tulu_oc',
            max_out_len=1024,
            batch_size=1,
            model_kwargs=dict(device_map='auto'),
            run_cfg=dict(num_gpus=1, num_procs=1),
        ),
        dict(
            type=HuggingFacewithChatTemplate,
            abbr='tulu_replicate_oc',
            max_out_len=1024,
            batch_size=2,
            model_kwargs=dict(device_map='auto'),
            run_cfg=dict(num_gpus=1, num_procs=1),
        ),
    ]
    ```
   To evaluate your own models, please directly add a new model dict with a different abbr.
   **Noted that name of the converted data should be the same as the abbr name listed in these two configs**
   Then, the opencompass will automatically locate the predictions during evaluation.
