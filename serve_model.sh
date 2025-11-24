source ./vllm/bin/activate

MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct-FP8"
MODEL_NAME_API="Qwen3-VL-4B"
PORT=8145
HOST=0.0.0.0
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=8192

echo "Запуск vLLM сервера..."
echo "Модель: $MODEL_NAME"
echo "Порт:   $PORT"
echo "Имя:    $MODEL_NAME_API"

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --port "$PORT" \
  --host "$HOST" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --served-model-name "$MODEL_NAME_API" \
> vllm4.log 2>&1 &
