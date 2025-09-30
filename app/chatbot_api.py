from app.chatbot_route import router
from pipeline.workflow import build_graph

from dotenv import load_dotenv
load_dotenv()

import os, uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Initializing FinDeep AI pipeline...")
        graph = build_graph(model_name = 'gpt-4o-mini',
                            embedding_model = "sentence-transformers/all-MiniLM-L6-v2")
        router.graph = graph
        print("FinDeep AI pipeline initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize AI pipeline: {e}")
        router.graph = None
    yield

app = FastAPI(
    title = "FinDeep services/chatbot API",
    lifespan = lifespan,
    docs_url = "/",
    redoc_url = None,
    openapi_url = "/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
async def read_root():
    return {"message": "FinDeep services/chatbot API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "graph_initialized": hasattr(router, 'graph') and router.graph is not None
    }

if __name__ == "__main__":
    chatbot_service_port = int(os.getenv("CHATBOT_SERVICE_PORT"))
    uvicorn.run(
        "services.chatbot.app.main:app",
        host = "0.0.0.0",
        port = chatbot_service_port,
        reload = True
    )