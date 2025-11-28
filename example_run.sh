#!/bin/bash

# Example 1: Training on specific GPUs (GPU 2 and 3)
# Use CUDA_VISIBLE_DEVICES to select specific GPUs
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes 2 main.py \
  --model_name_or_path google/gemma-3-270m-it \
  --dataset_name Exploration-Lab/CS779-Fall25 \
  --dataset_config_name Assignment-3-em \
  --text_column text \
  --output_dir ./outputs/gemma-270m-wikitext \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 500 \
  --max_seq_length 2048 \
  --bf16 \
  --gradient_checkpointing \
  --save_total_limit 3 \
  --report_to tensorboard

# Example 2: Training with local JSONL files
# python main.py \
#   --model_name_or_path google/gemma-270m \
#   --train_file ./data/train.jsonl \
#   --validation_file ./data/val.jsonl \
#   --text_column text \
#   --output_dir ./outputs/gemma-270m-custom \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 8 \
#   --learning_rate 2e-5 \
#   --bf16 \
#   --gradient_checkpointing

# Example 3: Multi-GPU training with all 4 GPUs
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 main.py \
#   --model_name_or_path google/gemma-270m \
#   --dataset_name your_dataset \
#   --output_dir ./outputs/gemma-270m-multigpu \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 16 \
#   --bf16 \
#   --gradient_checkpointing

# Example 4: Multi-node training (run on each node)
# accelerate launch \
#   --num_machines 2 \
#   --num_processes 8 \
#   --machine_rank 0 \
#   --main_process_ip YOUR_MAIN_NODE_IP \
#   --main_process_port 29500 \
#   main.py \
#   --model_name_or_path google/gemma-270m \
#   --dataset_name your_dataset \
#   --output_dir ./outputs/gemma-270m-multinode \
#   --per_device_train_batch_size 2 \
#   --bf16
