# 全局 ARG（在所有 FROM 之前声明）
ARG MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
ARG BASE_IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.3

# 第一阶段：下载模型
FROM python:3.12-slim AS downloader

ARG MODEL_ID

RUN pip install --no-cache-dir "huggingface_hub[cli]"

# 使用 huggingface-cli 下载模型到默认缓存目录
# 这会创建标准的 HF 缓存结构：~/.cache/huggingface/hub/models--org--model/...
ENV HF_HOME=/data
RUN hf download ${MODEL_ID}

# 第二阶段：TEI 运行时
FROM ${BASE_IMAGE}

ARG MODEL_ID

# 设置离线模式，启动时不会尝试联网检查/下载模型
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# 从第一阶段复制已下载的模型（使用标准 HF 缓存结构）
COPY --from=downloader /data/hub /data

# 设置模型 ID 环境变量，TEI 会自动读取
ENV MODEL_ID=${MODEL_ID}
