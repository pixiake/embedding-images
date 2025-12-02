#!/bin/bash
# Reranker 服务快速测试脚本

set -e

echo "================================"
echo "Qwen3-Reranker 服务测试"
echo "================================"

# 检查服务是否运行
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ 错误：服务未运行"
    echo "请先启动服务："
    echo "  docker run -d -p 8000:8000 reranker-service:latest"
    echo "或："
    echo "  python reranker_server.py"
    exit 1
fi

echo "✓ 服务运行中"
echo ""

# 测试健康检查
echo "1. 测试健康检查..."
curl -s http://localhost:8000/health | python3 -m json.tool
echo ""

# 测试 rerank 接口
echo "2. 测试 rerank 接口..."
curl -s -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是深度学习?",
    "documents": [
      "深度学习是机器学习的一个子集，使用多层神经网络。",
      "今天天气真好。",
      "神经网络受到人脑的启发。",
      "Python 是一种流行的编程语言。"
    ],
    "top_n": 2
  }' | python3 -m json.tool

echo ""
echo "================================"
echo "✅ 测试完成"
echo "================================"
