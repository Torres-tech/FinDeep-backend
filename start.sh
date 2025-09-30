#!/bin/bash

# FinDeep Backend Startup Script

echo "🚀 Starting FinDeep Backend..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating a template..."
    echo "# FinDeep Backend Environment Configuration

# Server Configuration
CHATBOT_SERVICE_PORT=8001

# OpenAI Configuration (required for the AI pipeline)
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Other AI model configurations
# MODEL_NAME=gpt-4o-mini
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2" > .env
    echo "📝 Please edit .env file and add your OpenAI API key"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Start the server
echo "🌟 Starting FinDeep Backend server..."
uvicorn app.chatbot_api:app --host 0.0.0.0 --port 8001 --reload

echo "✅ FinDeep Backend is running on http://localhost:8001"
echo "📖 API Documentation: http://localhost:8001/"
