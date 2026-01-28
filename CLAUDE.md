# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements distributed fine-tuning for Google's Gemma-270M causal language model using HuggingFace Transformers and Accelerate. The architecture supports single-GPU, multi-GPU, and multi-node training configurations.

## Repository Structure

**Core Files:**
- `main.py`: Primary training script built on HuggingFace Trainer API
- `requirements.txt`: Python dependencies (torch, transformers, datasets, accelerate, etc.)
- `train.py`: Empty placeholder file

**Helper Scripts:**
- `example_run.sh`: Comprehensive examples for different training scenarios
- `run_node0.sh`: Pre-configured script for main node in multi-node setup
- `run_node1.sh`: Pre-configured script for worker node in multi-node setup
- `multinode_train.sh`: Documentation and examples for multi-node training
- `verify_setup.sh`: Verification script to check environment, GPUs, and dependencies

## Core Architecture

**main.py**: The primary training script built on HuggingFace Trainer API. Key components:
- `load_and_prepare_dataset()`: Handles both HuggingFace datasets and local files (JSON/JSONL/CSV/TXT), tokenizes, and groups texts into fixed-length chunks
- `parse_args()`: Extensive CLI argument parsing for model, data, and training configuration
- Uses `DataCollatorForLanguageModeling` with `mlm=False` for causal language modeling
- Automatic checkpoint detection and resumption via `get_last_checkpoint()`
- The script tokenizes raw text, concatenates examples, and splits into `max_seq_length` chunks (default 2048 tokens)

**Dataset Processing Pipeline:**
1. Load raw dataset (HuggingFace hub or local files)
2. Tokenize all texts with truncation at `max_seq_length`
3. Concatenate all tokenized sequences
4. Split into fixed-length chunks of exactly `max_seq_length` tokens
5. Drop small remainder that doesn't fill a complete chunk

**Dataset Expectations**: All datasets must contain a text column (default: "text", configurable via `--text_column`)

**Distributed Training**: Uses HuggingFace Accelerate. The Trainer automatically handles distributed logic when launched with `accelerate launch`.

## Environment Setup

**Hardware**: Designed for NVIDIA A40 GPUs (48GB VRAM each). Default configuration uses GPUs 2 and 3.

**Installation:**
```bash
pip install -r requirements.txt
```

**Optional Setup:**
```bash
accelerate config  # Interactive configuration for distributed training
```

**Verification:**
```bash
bash verify_setup.sh
```
This script checks:
- Python and PyTorch versions
- CUDA availability and version
- GPU count and names
- Access to specific GPUs (2 and 3)
- Transformers, Accelerate, and Datasets versions
- Gemma model accessibility (requires HuggingFace token)
- Network configuration for multi-node setup
- Port 29500 availability

## Running Training

### Single GPU
```bash
python main.py --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --output_dir ./outputs --bf16
```

### Multi-GPU (specific GPUs)
```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 main.py \
  --dataset_name wikitext --output_dir ./outputs --bf16 --gradient_checkpointing
```

### Multi-Node Training

**Quick Start:**
```bash
# On Node 0 (Main Node)
bash run_node0.sh

# On Node 1 (Worker Node)
bash run_node1.sh
```

**Before running**, edit the scripts to configure:
- `MAIN_NODE_IP`: Node 0's IP address (default: "192.168.1.100")
- `GPUS`: Which GPUs to use (default: "2,3")
- `TOTAL_GPUS`: Total across all nodes (default: 4 for 2 nodes × 2 GPUs)

**Key Multi-Node Requirements:**
- Set `MAIN_NODE_IP` to Node 0's IP address
- `GPUS` environment variable independently selects GPUs on each node
- `TOTAL_GPUS` must equal total GPUs across ALL nodes
- `--num_processes` must match `TOTAL_GPUS`
- `--machine_rank`: 0 for main node, 1 for second node, 2 for third, etc.
- All nodes must use **identical** training arguments
- Output directory should be on **shared storage** accessible to all nodes (e.g., `/shared/outputs/`)
- Port 29500 must be open and accessible between all nodes
- All nodes must have the same code version and dependencies

**Manual Multi-node example:**
```bash
# Node 0
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_machines 2 --num_processes 4 --machine_rank 0 \
  --main_process_ip 192.168.1.100 --main_process_port 29500 --multi_gpu main.py \
  --dataset_name wikitext --output_dir /shared/outputs/gemma-270m --bf16

# Node 1
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_machines 2 --num_processes 4 --machine_rank 1 \
  --main_process_ip 192.168.1.100 --main_process_port 29500 --multi_gpu main.py \
  --dataset_name wikitext --output_dir /shared/outputs/gemma-270m --bf16
```

**Common Configurations:**
- 2 nodes × 2 GPUs: `GPUS="2,3"`, `TOTAL_GPUS=4`, `--num_processes 4`
- 2 nodes × 4 GPUs: `GPUS="0,1,2,3"`, `TOTAL_GPUS=8`, `--num_processes 8`
- 3 nodes × 2 GPUs: `GPUS="2,3"`, `TOTAL_GPUS=6`, `--num_processes 6`

## Custom Datasets

For local datasets:
```bash
# JSONL format
python main.py --train_file data/train.jsonl --validation_file data/val.jsonl \
  --text_column text --output_dir ./outputs

# CSV format
python main.py --train_file data/train.csv --text_column text --output_dir ./outputs
```

Dataset format: `{"text": "your training text here"}`

## Key Training Parameters

**Model:**
- `--model_name_or_path`: Model ID (default: "google/gemma-270m")
- `--use_flash_attention`: Enable Flash Attention 2 (requires flash-attn package, not in default requirements)

**Data:**
- `--dataset_name`: HuggingFace dataset name (e.g., "wikitext")
- `--dataset_config_name`: Dataset configuration (e.g., "wikitext-2-raw-v1")
- `--train_file`: Path to local training file (JSON/JSONL/CSV/TXT)
- `--validation_file`: Path to local validation file
- `--text_column`: Column name containing text (default: "text")
- `--max_seq_length`: Maximum sequence length (default: 2048)

**Training:**
- `--output_dir`: Output directory for checkpoints (default: "./outputs")
- `--num_train_epochs`: Number of epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per GPU (default: 4, can increase to 8-16 for A40s)
- `--per_device_eval_batch_size`: Evaluation batch size per GPU (default: 4)
- `--gradient_accumulation_steps`: Accumulation steps (default: 8). Effective batch size = batch_size × accumulation × num_gpus
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay (default: 0.01)
- `--warmup_steps`: Warmup steps (default: 500)

**Memory Optimization:**
- `--bf16`: Use bfloat16 mixed precision (recommended for A40s)
- `--fp16`: Use float16 mixed precision (alternative to bf16)
- `--gradient_checkpointing`: Reduces memory usage at ~20% speed cost

**Logging & Checkpointing:**
- `--logging_steps`: Log every N steps (default: 10)
- `--save_steps`: Save checkpoint every N steps (default: 500)
- `--eval_steps`: Evaluate every N steps (default: 500, only if validation data provided)
- `--save_total_limit`: Max checkpoints to keep (default: 3)
- `--report_to`: Logging tool - "tensorboard", "wandb", or "none" (default: "tensorboard")
- `--resume_from_checkpoint`: Path to resume from specific checkpoint

**Miscellaneous:**
- `--seed`: Random seed (default: 42)

## Troubleshooting

**Multi-Node Issues:**

**Nodes can't communicate:**
- Verify nodes can ping each other: `ping <NODE_IP>`
- Check firewall allows port 29500: `sudo ufw allow 29500`
- Test connection: `nc -vz <MAIN_NODE_IP> 29500`

**Training hangs at initialization:**
- Ensure all nodes use same `num_machines` and `num_processes`
- Verify `machine_rank` is unique on each node (0, 1, 2, ...)
- Check all nodes have identical code and dependencies
- Ensure all training arguments are identical across nodes

**Different GPU counts per node:**
- Calculate total: sum of all GPUs across all nodes
- Example: Node0 (2 GPUs) + Node1 (2 GPUs) = `--num_processes 4`
- Example: Node0 (4 GPUs) + Node1 (4 GPUs) = `--num_processes 8`
- Use `CUDA_VISIBLE_DEVICES` to select which GPUs on each node

**Out of memory (OOM):**
- Reduce `--per_device_train_batch_size` (try 4 → 2 → 1)
- Increase `--gradient_accumulation_steps` to maintain effective batch size
- Enable `--gradient_checkpointing`
- Reduce `--max_seq_length` (try 2048 → 1024 → 512)
- Disable flash attention if enabled

**Checkpoint issues:**
- Checkpoints auto-resume from `output_dir` by default
- Override with `--resume_from_checkpoint <path>`
- Checkpoints saved every `--save_steps` with `--save_total_limit` most recent kept

## Important Implementation Notes

- The script automatically sets `tokenizer.pad_token = tokenizer.eos_token` if pad token not defined
- Gradient checkpointing requires `use_cache=False` in model config (handled automatically)
- Flash Attention 2 requires `torch_dtype=torch.bfloat16` (set automatically when enabled)
- Multi-node training requires network connectivity on port 29500 between all nodes
- Training metrics logged to TensorBoard by default in `{output_dir}/runs/`
- Evaluation only runs if `--validation_file` or validation split exists in dataset
- The `group_texts()` function drops small remainders that don't fill complete chunks
- Model embeddings auto-resize if tokenizer vocab exceeds embedding size
- Use comments after every line of code to describe what that line does..
