#!/usr/bin/env python3
"""
Qwen3-Reranker FastAPI 服务
基于官方推荐的 transformers 方式实现
支持 OpenAI 兼容的 /rerank 接口
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen3-Reranker Service",
    description="Text reranking service using Qwen3-Reranker-0.6B",
    version="1.0.0"
)

# 全局变量
model = None
tokenizer = None
token_true_id = None
token_false_id = None
prefix_tokens = None
suffix_tokens = None
MAX_LENGTH = 8192

class RerankRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    documents: Optional[List[str]] = Field(None, description="待排序的文档列表")
    texts: Optional[List[str]] = Field(None, description="待排序的文本列表（与documents等效）")
    top_n: Optional[int] = Field(None, description="返回前N个结果")
    return_documents: bool = Field(True, description="是否返回文档内容")
    instruction: Optional[str] = Field(
        None, 
        description="自定义指令（默认为通用检索指令）"
    )

class RerankResult(BaseModel):
    index: int = Field(..., description="原始文档在输入列表中的索引")
    relevance_score: float = Field(..., description="相关性分数，范围 0-1")

class RerankResponse(BaseModel):
    id: str = Field(default="", description="请求 ID")
    results: List[RerankResult] = Field(..., description="重排序结果列表")
    meta: Dict[str, Any] = Field(default_factory=dict, description="元数据信息")

def format_instruction(instruction: str, query: str, doc: str) -> str:
    """格式化输入"""
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def process_inputs(pairs: List[str]):
    """预处理输入"""
    inputs = tokenizer(
        pairs, 
        padding=False, 
        truncation='longest_first',
        return_attention_mask=False, 
        max_length=MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens)
    )
    
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=MAX_LENGTH)
    
    # 移动到模型设备
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    
    return inputs

@torch.no_grad()
def compute_scores(inputs) -> List[float]:
    """计算相关性分数"""
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global model, tokenizer, token_true_id, token_false_id, prefix_tokens, suffix_tokens
    
    model_path = os.getenv("MODEL_PATH", "Qwen/Qwen3-Reranker-0.6B")
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        logger.info("✓ Tokenizer loaded")
        
        # 自动检测设备
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"🎮 GPU detected: {gpu_name} (count: {gpu_count})")
            logger.info(f"Using device: {device}")
            
            # GPU 模式：使用 float16 和 device_map
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            ).eval()
        else:
            device = "cpu"
            logger.info("💻 No GPU detected, using CPU")
            logger.info(f"Using device: {device}")
            
            # CPU 模式：使用 float32
            model = AutoModelForCausalLM.from_pretrained(model_path).eval()
        
        logger.info(f"✓ Model loaded on {device}")
        logger.info(f"  Model type: {model.__class__.__name__}")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        
        # 准备特殊 tokens
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        
        # 准备 prefix 和 suffix
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        
        logger.info("✓ Service ready")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "Qwen3-Reranker Service",
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "version": "1.0.0",
        "endpoints": ["/health", "/rerank", "/v1/rerank"]
    }

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "device": str(model.device) if model else "not loaded"
    }

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """重排序接口"""
    try:
        # 获取文档列表
        documents = request.documents or request.texts
        if not documents:
            raise HTTPException(
                status_code=400, 
                detail="Either 'documents' or 'texts' field is required"
            )
        
        if len(documents) == 0:
            raise HTTPException(status_code=400, detail="Documents list cannot be empty")
        
        # 格式化输入
        instruction = request.instruction
        pairs = [
            format_instruction(instruction, request.query, doc) 
            for doc in documents
        ]
        
        # 处理输入
        inputs = process_inputs(pairs)
        
        # 计算分数
        scores = compute_scores(inputs)
        
        # 构建结果
        results = [
            RerankResult(
                index=i,
                relevance_score=score
            )
            for i, score in enumerate(scores)
        ]
        
        # 按分数排序
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 限制返回数量
        if request.top_n:
            results = results[:request.top_n]
        
        # 生成请求 ID
        import uuid
        request_id = f"rerank-{uuid.uuid4().hex[:16]}"
        
        return RerankResponse(
            id=request_id,
            results=results,
            meta={
                "api_version": {"version": "1"},
                "billed_units": {"search_units": len(documents)}
            }
        )
        
    except Exception as e:
        logger.error(f"Rerank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/rerank", response_model=RerankResponse)
async def v1_rerank(request: RerankRequest):
    """OpenAI 兼容的重排序接口"""
    return await rerank(request)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
