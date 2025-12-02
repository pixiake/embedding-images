# embedding-images

文本嵌入和重排序服务镜像，支持 Embedding 和 Reranker 两种服务。

## 项目说明

本项目提供两种服务：

1. **Embedding 服务** - 基于 [Hugging Face TEI](https://github.com/huggingface/text-embeddings-inference)，高性能文本向量化
2. **Reranker 服务** - 基于 Qwen3-Reranker，使用 transformers + FastAPI 实现，支持文本重排序

## 特性

### Embedding 服务 (TEI)
- 🚀 高性能：基于 Rust 实现，支持动态批处理
- 🔌 API 兼容：支持 OpenAI 兼容的 `/v1/embeddings` 接口
- 📦 开箱即用：无需编写代码，直接使用

### Reranker 服务 (Qwen3-Reranker)
- 🎯 高精度：使用 Qwen3-Reranker-0.6B 模型
- 💻 CPU 支持：支持 CPU 和 GPU 运行
- 🔌 标准接口：提供 `/rerank` 和 `/v1/rerank` 接口
- 📊 相关性评分：返回 0-1 之间的相关性分数

---

## Embedding 服务

### 构建镜像

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

响应示例：
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.1, 0.2, ...], "index": 0},
    {"object": "embedding", "embedding": [0.3, 0.4, ...], "index": 1}
  ],
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
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

### Embedding 模型

| 模型 | 维度 | 说明 |
|------|------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 轻量级，速度快 |
| `BAAI/bge-small-en-v1.5` | 384 | 英文小模型 |
| `BAAI/bge-base-en-v1.5` | 768 | 英文基础模型 |
| `BAAI/bge-large-en-v1.5` | 1024 | 英文大模型 |
| `BAAI/bge-m3` | 1024 | 多语言模型 |

---

## Reranker 服务

### 特性

- ✅ **自动检测 CPU/GPU**：镜像同时支持 CPU 和 GPU 环境，运行时自动检测硬件
- ✅ **一个镜像通用**：无需构建不同的 CPU/GPU 版本
- ✅ **官方实现**：使用 Qwen3-Reranker 官方推荐的实现方式

### 构建 Reranker 镜像

默认使用 **Qwen3-Reranker-0.6B** 模型。

```bash
# 构建通用镜像（支持 CPU 和 GPU）
docker build -f Dockerfile.reranker -t reranker-service:latest .
```

#### 使用其他模型
```bash
docker build -f Dockerfile.reranker \
  --build-arg MODEL_ID=Qwen/Qwen3-Reranker-4B \
  -t reranker-service:4b .
```

### 运行 Reranker 容器

```bash
# CPU 运行（默认端口 8000）
docker run -p 8000:8000 reranker-service:latest

# GPU 运行（需要安装 nvidia-docker）
docker run --gpus all -p 8000:8000 reranker-service:latest

# 指定端口
docker run -p 8001:8000 reranker-service:latest

# GPU 运行（需要安装 nvidia-docker）
# 注意：需要修改 Dockerfile 中的 torch 安装为 GPU 版本
docker run --gpus all -p 8000:8000 reranker-service:latest
```

### Reranker API 使用

#### 健康检查
```bash
curl http://localhost:8000/health
```

响应：
```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "device": "cpu"
}
```

#### 重排序文档
```bash
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Deep Learning?",
    "documents": [
      "Deep Learning is a subset of machine learning that uses neural networks.",
      "The weather is nice today.",
      "Neural networks are inspired by the human brain."
    ],
    "top_n": 2
  }'
```

响应示例（Cohere Rerank API 兼容格式）：
```json
{
  "id": "rerank-a3ef62fab5714b48",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.9995
    },
    {
      "index": 2,
      "relevance_score": 0.0343
    }
  ],
  "meta": {
    "api_version": {"version": "1"},
    "billed_units": {"search_units": 3}
  }
}
```

#### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 搜索查询 |
| `documents` | array | 是* | 待排序的文档列表 |
| `texts` | array | 是* | 待排序的文本列表（与 documents 等效） |
| `top_n` | int | 否 | 返回前 N 个结果 |
| `return_documents` | bool | 否 | 是否返回文档内容（默认 true） |
| `instruction` | string | 否 | 自定义指令 |

*注：`documents` 和 `texts` 二选一

### 支持的 Reranker 模型

| 模型 | 参数量 | 说明 | 支持 |
|------|--------|------|------|
| `Qwen/Qwen3-Reranker-0.6B` | 0.6B | 轻量高效，推荐 | ✅ CPU/GPU |
| `Qwen/Qwen3-Reranker-4B` | 4B | 更高精度 | ✅ CPU/GPU |
| `Qwen/Qwen3-Reranker-8B` | 8B | 最高精度 | ✅ GPU |

**注意**：本 Reranker 服务使用 Qwen3-Reranker 系列模型，这些模型基于 CausalLM 架构，与 TEI 的 Sequence Classification reranker 不兼容。

## 参考

- [TEI GitHub](https://github.com/huggingface/text-embeddings-inference)
- [Qwen3-Reranker 模型](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [TEI 文档](https://huggingface.co/docs/text-embeddings-inference)