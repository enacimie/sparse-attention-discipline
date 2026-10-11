#!/bin/bash
# Runner: evals (3 configs) + compile llama.cpp CUDA, en paralelo
cd /root/mlattention
nohup bash -c 'python bench_eval.py --config q4 --n 300 > /root/evals_q4.log 2>&1 && python bench_eval.py --config q4_sparse --n 300 > /root/evals_q4s.log 2>&1 && python bench_eval.py --config 16bit --n 300 > /root/evals_16.log 2>&1 && echo EVALS_DONE' > /root/evals_chain.log 2>&1 &

cd /root
rm -rf llama.cpp
nohup bash -c 'git clone --depth 1 https://github.com/ggml-org/llama.cpp llama.cpp && cd llama.cpp && export PATH=/usr/local/cuda-12.4/bin:$PATH && cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 > /root/cmake.log 2>&1 && cmake --build build --config Release -j 16 > /root/make.log 2>&1 && echo COMPILE_DONE > /root/compile_done.flag' > /root/compile_chain.log 2>&1 &

echo "LANZADO: evals + compile en paralelo"
