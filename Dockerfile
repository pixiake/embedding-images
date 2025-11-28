# 使用 Hugging Face Text Embeddings Inference (TEI) 作为基础镜像
# 支持通过 ARG 传递基础镜像（CPU/GPU 版本）
ARG BASE_IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.3

FROM ${BASE_IMAGE}

# 设置模型名称（构建时传入）
ARG MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
# 将 ARG 转为 ENV，使其在运行时可用
ENV MODEL_ID=${MODEL_ID}

# TEI 默认端口为 80，这里改为 8000 保持兼容
ENV PORT=8000

# 预下载模型到镜像中（可选，如果希望离线使用）
# 注意：TEI 镜像中使用 text-embeddings-router 来下载模型
# 如果不预下载，启动时会自动下载

EXPOSE 8000

# TEI 启动命令
# 使用 shell form 以便能够展开环境变量
# --model-id: 模型名称（使用构建时传入的 MODEL_ID）
# --port: 服务端口
# --hostname: 监听地址
ENTRYPOINT []
CMD text-embeddings-router --model-id ${MODEL_ID} --port 8000 --hostname 0.0.0.0
