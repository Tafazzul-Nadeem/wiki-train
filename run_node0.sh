#!/bin/bash
# Run this on Node 0 (Main Node)

# Configuration - MODIFY THESE
MAIN_NODE_IP="192.168.1.100"  # Replace with Node 0's IP address
GPUS="2,3"  # Which GPUs to use on this node
NUM_GPUS=2  # Number of GPUs on this node
TOTAL_GPUS=4  # Total GPUs across all nodes (e.g., 2 nodes × 2 GPUs = 4)

# Training command
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
  --num_machines 2 \
  --num_processes $TOTAL_GPUS \
  --machine_rank 0 \
  --main_process_ip $MAIN_NODE_IP \
  --main_process_port 29500 \
  --multi_gpu \
  main.py \
  --model_name_or_path google/gemma-270m \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir /shared/outputs/gemma-270m-multinode \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --bf16 \
  --gradient_checkpointing \
  --save_steps 500 \
  --eval_steps 500 \
  --logging_steps 10 \
  --save_total_limit 3 \
  --report_to tensorboard
