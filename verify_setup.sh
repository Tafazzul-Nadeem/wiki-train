#!/bin/bash

echo "=== GPU Training Setup Verification ==="
echo ""

# Check Python version
echo "1. Python version:"
python --version
echo ""

# Check PyTorch and CUDA
echo "2. PyTorch and CUDA:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

# List all GPUs
echo "3. Available GPUs:"
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

# Check specific GPUs (2 and 3)
echo "4. Testing access to GPU 2 and 3:"
CUDA_VISIBLE_DEVICES=2,3 python -c "import torch; print(f'Visible GPUs: {torch.cuda.device_count()}'); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

# Check transformers
echo "5. Transformers version:"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
echo ""

# Check accelerate
echo "6. Accelerate version:"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
echo ""

# Check datasets
echo "7. Datasets version:"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"
echo ""

# Test Gemma model access (requires HF token)
echo "8. Testing Gemma model access:"
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('google/gemma-270m', trust_remote_code=True); print('✓ Gemma model accessible')" 2>&1 | head -n 5
echo ""

# Check network connectivity (for multi-node)
echo "9. Network info (for multi-node):"
hostname -I 2>/dev/null || ipconfig getifaddr en0 2>/dev/null || echo "Could not determine IP"
echo ""

# Check if port 29500 is available
echo "10. Checking if port 29500 is available:"
(nc -z localhost 29500 && echo "Port 29500 is in use") || echo "Port 29500 is available"
echo ""

echo "=== Verification Complete ==="
