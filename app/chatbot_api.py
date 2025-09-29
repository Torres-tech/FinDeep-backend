from app.chatbot_route import router
from pipeline.workflow import build_graph

from dotenv import load_dotenv
load_dotenv()

import os, uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    graph = build_graph(model_name = 'gpt-4o-mini',
                        embedding_model = "sentence-transformers/all-MiniLM-L6-v2")
    router.graph = graph
    yield

app = FastAPI(
    title = "FinDeep services/chatbot API",
    lifespan = lifespan,
    docs_url = "/",
    redoc_url = None,
    openapi_url = "/openapi.json"
)

app.include_router(router)

@app.get("/")
async def read_root():
    return {"message": "FinDeep services/chatbot API is running"}

if __name__ == "__main__":
    chatbot_service_port = int(os.getenv("CHATBOT_SERVICE_PORT"))
    uvicorn.run(
        "services.chatbot.app.main:app",
        host = "0.0.0.0",
        port = chatbot_service_port,
        reload = True
    )