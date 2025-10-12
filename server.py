from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
import torch
import uvicorn
import os

app = FastAPI()

# 支持通过环境变量传递模型名称
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# 自动检测GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].to("cpu").tolist()

class EmbeddingRequest(BaseModel):
    input: List[str]
    model: Optional[str] = None

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embedding(req: EmbeddingRequest):
    # 兼容OpenAI API格式
    if not req.input:
        raise HTTPException(status_code=400, detail="Input cannot be empty.")
    # OpenAI支持input为str或str数组
    inputs = req.input if isinstance(req.input, list) else [req.input]
    data = []
    total_tokens = 0
    for idx, text in enumerate(inputs):
        emb = get_embedding(text)
        tokens = len(tokenizer.tokenize(text))
        total_tokens += tokens
        data.append({"object": "embedding", "embedding": emb, "index": idx})
    usage = {"prompt_tokens": total_tokens, "total_tokens": total_tokens}
    return EmbeddingResponse(data=data, model=MODEL_NAME, usage=usage)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
