export WANDB_PROJECT="Block-Attention"

DS_CONFIG="configs/ds_stage2.json"

MODEL_NAME="/root/hubs/meta-llama/Llama-3.1-8B"

TRAIN_FP="datahub/tulu3/sft.train"
# You can create a test set by yourself.
EVAL_FP=""

LEARNING_RATE=5e-6

SAVE_DIR=""
RUN_NAME=""

mkdir -p $SAVE_DIR

SCRIPT_PATH=$(readlink -f "$0")

cp $(realpath $EVAL_FP) $SAVE_DIR
cp $(realpath $TRAIN_FP) $SAVE_DIR
cp $SCRIPT_PATH $SAVE_DIR


deepspeed --num_gpus 8 trainer/hf_trainer.py \
  --model_name $MODEL_NAME \
  --train_fp $TRAIN_FP \
  --eval_fp $EVAL_FP \
  --dataloader_num_workers 8 \
  --dataloader_prefetch_factor 32 \
  --do_train \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate $LEARNING_RATE \
  --weight_decay 0.0 \
  --lr_scheduler_type "linear" \
  --loss_reduction "sum" \
  --warmup_ratio 0.03 \
  --num_train_epochs 2 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --eval_strategy "steps" \
  --eval_steps 500 \
  --logging_steps 1 \
  --bf16 \
  --optim "adamw_torch_fused" \
  --output_dir $SAVE_DIR \
  --logging_dir $SAVE_DIR \
  --run_name $RUN_NAME \
  --report_to "wandb" \
  --deepspeed $DS_CONFIG
