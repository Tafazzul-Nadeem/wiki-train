# wiki-train
Multi-GPU and Multi-Node training for Gemma-270M.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Accelerate (choose your setup):
```bash
accelerate config
```

Or use the quick setup commands below.

## Quick Start

### Single GPU
```bash
python main.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir ./outputs/gemma-270m \
  --bf16 \
  --gradient_checkpointing
```

### Multi-GPU (Automatic)
```bash
accelerate launch main.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir ./outputs/gemma-270m \
  --bf16 \
  --gradient_checkpointing
```

### Multi-GPU (Specify number of GPUs)
```bash
accelerate launch --num_processes 4 main.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir ./outputs/gemma-270m \
  --bf16
```

### Train on Specific GPUs (e.g., GPU 2 and 3)
```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 main.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --output_dir ./outputs/gemma-270m \
  --bf16 \
  --gradient_checkpointing
```

### Multi-Node Training with Specific GPUs

**Setup:** 2 nodes, each using GPU 2 and 3 (4 GPUs total)

**On Node 0 (Main Node):**
```bash
bash run_node0.sh
```

**On Node 1 (Worker Node):**
```bash
bash run_node1.sh
```

**Or manually:**

**Node 0:**
```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --num_machines 2 \
  --num_processes 4 \
  --machine_rank 0 \
  --main_process_ip 192.168.1.100 \
  --main_process_port 29500 \
  --multi_gpu \
  main.py \
  --dataset_name wikitext \
  --output_dir /shared/outputs/gemma-270m
```

**Node 1:**
```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --num_machines 2 \
  --num_processes 4 \
  --machine_rank 1 \
  --main_process_ip 192.168.1.100 \
  --main_process_port 29500 \
  --multi_gpu \
  main.py \
  --dataset_name wikitext \
  --output_dir /shared/outputs/gemma-270m
```

**Important:**
- Replace `192.168.1.100` with your main node's actual IP
- Ensure port 29500 is open for communication between nodes
- Use shared storage for `--output_dir` accessible from all nodes
- `num_processes` = total GPUs across ALL nodes
- All nodes must use identical training arguments

## Dataset Format

Your dataset should have a `text` column. For custom datasets:
- JSONL: `{"text": "your text here"}`
- CSV: Column named "text"
- Use `--text_column` to specify different column name

See `example_run.sh` for more examples.

## Verification

Test your setup before training:
```bash
bash verify_setup.sh
```

## Multi-Node Troubleshooting

**Issue: Nodes can't communicate**
- Verify both nodes can ping each other
- Check firewall allows port 29500: `sudo ufw allow 29500`
- Test connection: `nc -vz <MAIN_NODE_IP> 29500`

**Issue: Training hangs at initialization**
- Ensure all nodes use the same `num_machines` and `num_processes`
- Verify `machine_rank` is unique (0, 1, 2, ...)
- Check all nodes have the same code and dependencies

**Issue: Different number of GPUs per node**
- Node 0 with 2 GPUs, Node 1 with 2 GPUs: `--num_processes 4`
- Node 0 with 4 GPUs, Node 1 with 4 GPUs: `--num_processes 8`
- CUDA_VISIBLE_DEVICES controls which GPUs are used on each node

**Issue: Out of memory**
- Reduce `--per_device_train_batch_size`
- Increase `--gradient_accumulation_steps`
- Enable `--gradient_checkpointing`
- Reduce `--max_seq_length`
