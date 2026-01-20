python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models/LLM-Research/Llama-3___2-3B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.65 \
    --trust-remote-code \
    --max-model-len 8192

python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models/Qwen/Qwen3-Embedding-0.6B \
    --port 8001 \
    --gpu-memory-utilization 0.2 \
    --trust-remote-code \
    --max-model-len 1024