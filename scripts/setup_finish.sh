#!/bin/bash
# Setup paralelo: descargas (modelo 52GB + GGUF 18GB) + torch + transformers + llama-cpp-python
nohup python3 /root/dl_qwen38.py > /root/dl.log 2>&1 &
nohup curl -L --retry 3 -C - -o /root/Qwen3.8-27B-Q4_K_M.gguf "https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf" > /root/dl_gguf.log 2>&1 &
nohup pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 > /root/pip_torch.log 2>&1 &
nohup pip install transformers accelerate datasets bitsandbytes scipy sentencepiece protobuf tokenizers huggingface_hub > /root/pip_rest.log 2>&1 &
nohup pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 > /root/pip_llama.log 2>&1 &
echo "LANZADO: descargas + torch + transformers + llama-cpp-python en paralelo"
