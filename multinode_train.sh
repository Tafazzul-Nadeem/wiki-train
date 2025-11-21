#!/bin/bash

# Multi-Node Training Script for Gemma-270M
# This script should be adapted and run on each node

# Configuration
MAIN_NODE_IP="192.168.1.100"  # Replace with your main node's IP address
MAIN_NODE_PORT=29500
NUM_MACHINES=2
TOTAL_PROCESSES=4  # Total processes across all nodes (e.g., 2 GPUs per node = 4 total)

# ============================================
# Node 0 (Main Node) - Using GPU 2 and 3
# ============================================
# Run this command on the main node:
#
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
#   --num_machines 2 \
#   --num_processes 4 \
#   --machine_rank 0 \
#   --main_process_ip 192.168.1.100 \
#   --main_process_port 29500 \
#   --multi_gpu \
#   main.py \
#   --model_name_or_path google/gemma-270m \
#   --dataset_name wikitext \
#   --dataset_config_name wikitext-2-raw-v1 \
#   --output_dir ./outputs/gemma-270m-multinode \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 4 \
#   --learning_rate 2e-5 \
#   --bf16 \
#   --gradient_checkpointing \
#   --save_steps 500 \
#   --logging_steps 10

# ============================================
# Node 1 (Worker Node) - Using GPU 2 and 3
# ============================================
# Run this command on the second node:
#
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
#   --num_machines 2 \
#   --num_processes 4 \
#   --machine_rank 1 \
#   --main_process_ip 192.168.1.100 \
#   --main_process_port 29500 \
#   --multi_gpu \
#   main.py \
#   --model_name_or_path google/gemma-270m \
#   --dataset_name wikitext \
#   --dataset_config_name wikitext-2-raw-v1 \
#   --output_dir ./outputs/gemma-270m-multinode \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 4 \
#   --learning_rate 2e-5 \
#   --bf16 \
#   --gradient_checkpointing \
#   --save_steps 500 \
#   --logging_steps 10

# ============================================
# Example: Using all 4 GPUs on each node
# ============================================
# Node 0:
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#   --num_machines 2 \
#   --num_processes 8 \
#   --machine_rank 0 \
#   --main_process_ip 192.168.1.100 \
#   --main_process_port 29500 \
#   main.py \
#   --dataset_name wikitext \
#   --output_dir ./outputs/gemma-270m-multinode \
#   --bf16

# Node 1:
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#   --num_machines 2 \
#   --num_processes 8 \
#   --machine_rank 1 \
#   --main_process_ip 192.168.1.100 \
#   --main_process_port 29500 \
#   main.py \
#   --dataset_name wikitext \
#   --output_dir ./outputs/gemma-270m-multinode \
#   --bf16

# ============================================
# Notes:
# ============================================
# 1. Replace MAIN_NODE_IP with your actual main node IP
# 2. Ensure both nodes can communicate on port 29500
# 3. CUDA_VISIBLE_DEVICES selects which GPUs to use on each node
# 4. num_processes = total number of GPUs across ALL nodes
# 5. machine_rank: 0 for main node, 1 for second node, etc.
# 6. All nodes must use the same training arguments
# 7. Make sure the output directory is accessible from all nodes (use shared storage)
