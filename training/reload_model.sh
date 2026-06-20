#!/bin/bash

BASE_MODEL="Qwen/Qwen2.5-32B-Instruct"
LORA_PATH="training/output_model"

echo "Restarting vLLM with LoRA..."

pkill -f vllm || true
sleep 5

${PY} -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL}" \
  --lora "${LORA_PATH}" \
  --trust-remote-code \
  --tokenizer "${BASE_MODEL}" \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  &
``