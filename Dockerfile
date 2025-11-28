# 全局 ARG（在所有 FROM 之前声明）
ARG MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
ARG BASE_IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.3

# 第一阶段：下载模型
FROM python:3.12-slim AS downloader

ARG MODEL_ID

RUN pip install --no-cache-dir "huggingface_hub[cli]"

# 下载模型到 /model 目录
RUN hf download ${MODEL_ID} --local-dir /model/${MODEL_ID}

# 第二阶段：TEI 运行时
FROM ${BASE_IMAGE}

ARG MODEL_ID

# 从第一阶段复制已下载的模型
COPY --from=downloader /model /model

# 使用本地路径，TEI 不会尝试下载
ENV MODEL_ID=/model/${MODEL_ID}