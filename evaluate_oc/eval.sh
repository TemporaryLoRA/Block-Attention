#!/bin/bash 

opencompass -m eval --models tulu \
    --datasets gsm8k_gen math_gen humaneval_gen mmlu_gen bbh_gen ifeval_gen drop_gen \
    --max-num-worker 100 \
    -r block_attention \
    --debug
