export TOKENIZERS_PARALLELISM="false"
export WANDB_PROJECT="Block-Attention"
DS_CONFIG="configs/ds_stage2.json"


# The path of Tulu3-SFT epoch 2 checkpoint
MODEL_NAME=""

TRAIN_FP="datahub/mix_tulu3_rag.train"
# You can create a test set by yourself.
EVAL_FP=""

LEARNING_RATE=2e-6

RUN_NAME=""
SAVE_DIR=""

mkdir -p $SAVE_DIR

SCRIPT_PATH=$(readlink -f "$0")

cp $(realpath $EVAL_FP) $SAVE_DIR
cp $(realpath $TRAIN_FP) $SAVE_DIR
cp $SCRIPT_PATH $SAVE_DIR


deepspeed --num_gpus 8 trainer/hf_trainer.py \
  --model_name $MODEL_NAME \
  --max_length 4096 \
  --train_fp $TRAIN_FP \
  --eval_fp $EVAL_FP \
  --dataloader_num_workers 1 \
  --dataloader_prefetch_factor 128 \
  --remove_unused_columns false \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing \
  --loss_reduction "sum" \
  --learning_rate $LEARNING_RATE \
  --weight_decay 0.0 \
  --lr_scheduler_type "linear" \
  --warmup_ratio 0.03 \
  --num_train_epochs 1 \
  --save_strategy "steps" \
  --save_steps 50 \
  --eval_strategy "steps" \
  --eval_steps 200 \
  --logging_steps 1 \
  --bf16 \
  --optim "adamw_torch_fused" \
  --output_dir $SAVE_DIR \
  --logging_dir $SAVE_DIR \
  --run_name  $RUN_NAME \
  --report_to "wandb" \
  --deepspeed $DS_CONFIG
