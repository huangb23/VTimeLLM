#!/bin/bash

MODEL_VERSION=chatglm3-6b
gpu_vis=0 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=29570


deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT vtimellm/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version plain \
    --data_path ./data/stage2_chinese.json \
    --feat_folder /path/to/stage2_feat \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin \
    --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage2 \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
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
