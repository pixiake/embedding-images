# 全局 ARG（在所有 FROM 之前声明）
ARG MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
ARG BASE_IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.3

# 第一阶段：下载模型
FROM python:3.12-slim AS downloader

ARG MODEL_ID

RUN pip install --no-cache-dir "huggingface_hub[cli]"

# 下载完整模型仓库到 /data/model 目录
RUN hf download ${MODEL_ID} --local-dir /data/model

# 第二阶段：TEI 运行时
FROM ${BASE_IMAGE}

# 模型缓存目录
ENV HF_HOME=/data
ENV HUGGINGFACE_HUB_CACHE=/data
# 设置离线模式，启动时不会尝试联网检查/下载模型
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# 从第一阶段复制已下载的模型
COPY --from=downloader /data/model /data/model

# TEI 默认端口为 80，这里改为 8000 保持兼容
ENV PORT=8000

EXPOSE 8000

# TEI 基础镜像已经设置了 ENTRYPOINT ["text-embeddings-router"]
# 使用本地模型路径而不是模型 ID
CMD ["--model-id", "/data/model", "--port", "8000", "--hostname", "0.0.0.0"]
