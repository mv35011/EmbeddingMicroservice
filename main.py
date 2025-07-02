import os
import tempfile
import asyncio
import aiohttp
from datetime import datetime
CACHE_DIR = os.path.join(os.getcwd(), "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR

from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
import torch
from pydantic import BaseModel

class EmbedRequest(BaseModel):
    texts: List[str]


class SingleEmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    embedding_model_name: str
    dimension: int


class SingleEmbedResponse(BaseModel):
    embedding: List[float]
    embedding_model_name: str
    dimension: int


class HealthResponse(BaseModel):
    status: str
    embedding_model_loaded: bool
    embedding_model_name: str


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model_info = {"loaded": False, "dimension": 384}
SELF_PING_ENABLED = os.environ.get("SELF_PING_ENABLED", "true").lower() == "true"
SELF_PING_INTERVAL = int(os.environ.get("SELF_PING_INTERVAL", "600"))
SELF_PING_URL = os.environ.get("SELF_PING_URL", "")


async def self_ping_task():
    """Background task to ping the service periodically to prevent it from sleeping"""
    if not SELF_PING_ENABLED or not SELF_PING_URL:
        print("Self-ping disabled or URL not configured")
        return

    print(f"Starting self-ping task - pinging every {SELF_PING_INTERVAL} seconds")

    while True:
        try:
            await asyncio.sleep(SELF_PING_INTERVAL)

            async with aiohttp.ClientSession() as session:
                ping_url = f"{SELF_PING_URL}/health"
                async with session.get(ping_url, timeout=30) as response:
                    if response.status == 200:
                        print(f"Self-ping successful at {datetime.now()}")
                    else:
                        print(f"Self-ping failed with status {response.status} at {datetime.now()}")

        except asyncio.CancelledError:
            print("Self-ping task cancelled")
            break
        except Exception as e:
            print(f"Self-ping error at {datetime.now()}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Loading model: {MODEL_NAME}...")
    ping_task = None
    if SELF_PING_ENABLED and SELF_PING_URL:
        ping_task = asyncio.create_task(self_ping_task())

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            local_files_only=False
        )
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            local_files_only=False
        )
        app.state.tokenizer = tokenizer
        app.state.model = model
        model_info["loaded"] = True
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model_info["loaded"] = False
        print("Trying alternative cache location...")
        try:
            alt_cache = "/tmp/hf_cache"
            os.makedirs(alt_cache, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir=alt_cache,
                local_files_only=False
            )
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                cache_dir=alt_cache,
                local_files_only=False
            )
            app.state.tokenizer = tokenizer
            app.state.model = model
            model_info["loaded"] = True
            print("Model loaded successfully with alternative cache")
        except Exception as e2:
            print(f"Error with alternative cache: {e2}")

    yield
    if ping_task:
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError:
            pass
    print("Shutting down...")


app = FastAPI(
    title="Embedding Service",
    description="FastAPI service for generating text embeddings using Hugging Face transformers",
    version="1.0.0",
    lifespan=lifespan
)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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
        embedding_model_loaded=model_info["loaded"],
        embedding_model_name=MODEL_NAME
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
            embedding_model_name=MODEL_NAME,
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
            embedding_model_name=MODEL_NAME,
            dimension=model_info["dimension"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")


@app.get("/ping-status")
async def ping_status():
    """Endpoint to check self-ping configuration"""
    return {
        "self_ping_enabled": SELF_PING_ENABLED,
        "self_ping_interval": SELF_PING_INTERVAL,
        "self_ping_url": SELF_PING_URL,
        "current_time": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    return {
        "message": "Embedding Service API",
        "model": MODEL_NAME,
        "model_loaded": model_info["loaded"],
        "cache_directory": CACHE_DIR,
        "self_ping_enabled": SELF_PING_ENABLED,
        "endpoints": {
            "health": "/health",
            "embed_multiple": "/embed",
            "embed_single": "/embed_single",
            "ping_status": "/ping-status",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)