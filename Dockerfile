# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

COPY server.py .

# 设置模型名称为环境变量（可根据需要修改）
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}

RUN pip install --no-cache-dir fastapi uvicorn transformers torch huggingface-hub
# 提前下载模型，保证离线可用
RUN huggingface-cli download $EMBEDDING_MODEL --local-dir /root/.cache/huggingface/hub

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
