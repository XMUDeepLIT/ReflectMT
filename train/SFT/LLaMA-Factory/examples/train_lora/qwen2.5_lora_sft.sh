#!/bin/bash

set -x
export request_id="run_$(date +%Y%m%d_%H%M%S)"

MODEL_PATH=

llamafactory-cli train \
    --model_name_or_path ${MODEL_PATH} \
    --stage sft \
    --do_train \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_target all \
    --dataset bc1 \
    --template default \
    --cutoff_len 2048 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir  \
    --logging_steps 10 \
    --save_steps 100 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000
