#!/bin/bash

MODEL_VERSION=chatglm3-6b
gpu_vis=0 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=29570


deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT vtimellm/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version plain \
    --data_path ./data/blip_laion_cc_sbu_558k_chinese.json \
    --feat_folder /path/to/stage1_feat \
    --tune_mm_mlp_adapter True \
    --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage1 \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
