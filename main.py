from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
import torch
import os
from pydantic import BaseModel
# run uvicorn main:app --reload



class EmbedRequest(BaseModel):
    texts: List[str]

class SingleEmbedRequest(BaseModel):
    text: str
class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str
    dimension: int
class SingleEmbedResponse(BaseModel):
    embedding: List[float]
    model_name: str
    dimension: int
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model_info = {"loaded": False, "dimension": 384}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"loading model:{MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        app.state.tokenizer = tokenizer
        app.state.model = model
        model_info["loaded"] = True
        print("model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model_info["loaded"] = False
    yield
    print("shutting down...")

app = FastAPI(title="Embedding Service",
              description="FastAPI service for generating text embeddings using Hugging Face transformers",
              version="1.0.0",
              lifespan=lifespan)
def mean_pooling(model_output, attention_mask):
    # this is for applying mean pooling to get sentence transformer
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded,1)/torch.clamp(input_mask_expanded.sum(1), min=1e-9)
def get_embeddings(texts: List[str], tokenizer, model) -> List[List[float]]:
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = mean_pooling(model_output, inputs['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy().tolist()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status='healthy' if model_info["loaded"] else 'unhealthy',
        model_loaded=model_info["loaded"],
        model_name=MODEL_NAME
    )
@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    if not model_info["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        embeddings = get_embeddings(
            request.texts,
            app.state.tokenizer,
            app.state.model
        )

        return EmbedResponse(
            embeddings=embeddings,
            model_name=MODEL_NAME,
            dimension=model_info["dimension"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")


@app.post("/embed_single", response_model=SingleEmbedResponse)
async def embed_single_text(request: SingleEmbedRequest):
    if not model_info["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        embeddings = get_embeddings(
            [request.text],
            app.state.tokenizer,
            app.state.model
        )

        return SingleEmbedResponse(
            embedding=embeddings[0],
            model_name=MODEL_NAME,
            dimension=model_info["dimension"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Embedding Service API",
        "model": MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "embed_multiple": "/embed",
            "embed_single": "/embed_single",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)