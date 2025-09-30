# Import the chat router that handles /chat endpoint
from app.chatbot_route import router
# Import the workflow builder for the original complex AI pipeline
from pipeline.workflow import build_graph

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import required modules for FastAPI server
import os, uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # For handling cross-origin requests
from contextlib import asynccontextmanager  # For application lifecycle management

# Application lifecycle manager - handles startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase - initialize AI pipeline
    try:
        print("Initializing FinDeep AI pipeline...")
        # Build the complex LangGraph workflow (original implementation)
        graph = build_graph(model_name = 'gpt-4o-mini',
                            embedding_model = "sentence-transformers/all-MiniLM-L6-v2")
        # Store the graph in the router for use in chat endpoint
        router.graph = graph
        print("FinDeep AI pipeline initialized successfully!")
    except Exception as e:
        # If pipeline initialization fails, set graph to None and continue
        print(f"Failed to initialize AI pipeline: {e}")
        router.graph = None
    yield  # Application runs here
    
    # Shutdown phase (currently empty, but could add cleanup code)

# Create FastAPI application instance
app = FastAPI(
    title = "FinDeep services/chatbot API",  # API title for documentation
    lifespan = lifespan,  # Use our custom lifespan manager
    docs_url = "/",  # Serve API docs at root URL
    redoc_url = None,  # Disable ReDoc documentation
    openapi_url = "/openapi.json"  # OpenAPI schema endpoint
)

# Add CORS middleware to allow frontend communication
# This is CRITICAL for React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,  # Allow cookies and authentication headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include the chat router to handle /chat endpoint
app.include_router(router)

# Root endpoint - basic API status check
@app.get("/")
async def read_root():
    return {"message": "FinDeep services/chatbot API is running"}

# Health check endpoint - monitors API and AI pipeline status
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",  # API is running
        "graph_initialized": hasattr(router, 'graph') and router.graph is not None  # AI pipeline status
    }

# Main entry point for running the server directly
if __name__ == "__main__":
    # Get port from environment variable
    chatbot_service_port = int(os.getenv("CHATBOT_SERVICE_PORT"))
    # Start the server with uvicorn
    uvicorn.run(
        "services.chatbot.app.main:app",  # App location (this might need updating)
        host = "0.0.0.0",  # Listen on all interfaces
        port = chatbot_service_port,  # Use configured port
        reload = True  # Auto-reload on code changes (development mode)
    )