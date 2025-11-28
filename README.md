# embedding-images

基于 [Hugging Face Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) 的文本嵌入服务镜像。

## 特性

- 🚀 高性能：基于 Rust 实现，支持动态批处理
- 🔌 API 兼容：支持 OpenAI 兼容的 `/v1/embeddings` 接口
- 📦 开箱即用：无需编写代码，直接使用

## 构建镜像

### CPU 版本
```bash
docker build -t embedding-service:cpu .
```

### GPU 版本 (CUDA)
```bash
docker build --build-arg BASE_IMAGE=ghcr.io/huggingface/text-embeddings-inference:1.5 -t embedding-service:gpu .
```

### 使用其他模型
```bash
docker build --build-arg MODEL_ID=BAAI/bge-small-en-v1.5 -t embedding-service:bge .
```

## 运行容器

### CPU 运行
```bash
docker run -p 8000:8000 embedding-service:cpu
```

### GPU 运行
```bash
docker run --gpus all -p 8000:8000 embedding-service:gpu
```

### 使用外部模型（不预下载到镜像）
```bash
docker run -p 8000:8000 \
  -v ~/.cache/huggingface:/data \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --port 8000
```

## API 使用

### 健康检查
```bash
curl http://localhost:8000/health
```

### 获取嵌入向量 (OpenAI 兼容格式)
```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello, world!", "How are you?"],
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }'
```

### TEI 原生格式
```bash
curl http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Hello, world!"]}'
```

## 可用的基础镜像

| 镜像标签 | 说明 |
|---------|------|
| `ghcr.io/huggingface/text-embeddings-inference:cpu-1.5` | CPU 版本 |
| `ghcr.io/huggingface/text-embeddings-inference:1.5` | CUDA 12 GPU 版本 |
| `ghcr.io/huggingface/text-embeddings-inference:turing-1.5` | CUDA 12 Turing GPU (T4, RTX 2000) |
| `ghcr.io/huggingface/text-embeddings-inference:89-1.5` | CUDA 12 Ampere 86 (A10, A40) |
| `ghcr.io/huggingface/text-embeddings-inference:hopper-1.5` | CUDA 12 Hopper (H100) |

## 推荐模型

| 模型 | 维度 | 说明 |
|------|------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 轻量级，速度快 |
| `BAAI/bge-small-en-v1.5` | 384 | 英文小模型 |
| `BAAI/bge-base-en-v1.5` | 768 | 英文基础模型 |
| `BAAI/bge-large-en-v1.5` | 1024 | 英文大模型 |
| `BAAI/bge-m3` | 1024 | 多语言模型 |

## 参考

- [TEI GitHub](https://github.com/huggingface/text-embeddings-inference)
- [TEI 文档](https://huggingface.co/docs/text-embeddings-inference)