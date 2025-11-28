# 全局 ARG（在所有 FROM 之前声明）
ARG MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
ARG BASE_IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.3

# 第一阶段：下载模型
FROM python:3.12-slim AS downloader

ARG MODEL_ID

RUN pip install --no-cache-dir huggingface_hub

# 下载模型到 /data 目录
# 使用环境变量传递 MODEL_ID，避免 shell 转义问题
ENV MODEL_ID=${MODEL_ID}
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['MODEL_ID'], cache_dir='/data')"

# 第二阶段：TEI 运行时
FROM ${BASE_IMAGE}

# 重新声明 ARG（每个 FROM 之后需要重新声明）
ARG MODEL_ID
# 将 ARG 转为 ENV，使其在运行时可用
ENV MODEL_ID=${MODEL_ID}

# 模型缓存目录（TEI 默认使用 /data）
ENV HF_HOME=/data
ENV HUGGINGFACE_HUB_CACHE=/data
# 设置离线模式，启动时不会尝试联网检查/下载模型
ENV HF_HUB_OFFLINE=1

# 从第一阶段复制已下载的模型
COPY --from=downloader /data /data

# TEI 默认端口为 80，这里改为 8000 保持兼容
ENV PORT=8000

EXPOSE 8000

# TEI 启动命令
# 使用 shell form 以便展开 $MODEL_ID 环境变量
CMD ["sh", "-c", "text-embeddings-router --model-id $MODEL_ID --port 8000 --hostname 0.0.0.0"]
