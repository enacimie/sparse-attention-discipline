#!/bin/bash
# Setup paralelo: descarga modelo + torch + transformers
nohup python3 /root/dl_qwen38.py > /root/dl.log 2>&1 &
nohup pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 > /root/pip_torch.log 2>&1 &
nohup pip install transformers accelerate datasets bitsandbytes scipy sentencepiece protobuf tokenizers huggingface_hub > /root/pip_rest.log 2>&1 &
echo "LANZADO: descarga + torch + transformers en paralelo"
