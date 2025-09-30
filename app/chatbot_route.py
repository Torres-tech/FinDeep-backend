# Import request/response schemas for type safety
from app.request_schema import ChatRequest, ChatResponse

# Import FastAPI components for API routing and error handling
from fastapi import APIRouter, HTTPException
# Import LangChain components (legacy - not used in simplified version)
from langchain_core.messages import HumanMessage, AIMessage
# Import OpenAI integration for direct AI calls
from langchain_openai import ChatOpenAI
import os  # For environment variable access

# Create API router for chat endpoints
router = APIRouter()

# Main chat endpoint - receives messages from frontend and returns AI responses
@router.post("/chat", response_model = ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        # Get OpenAI API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Validate API key - check if it exists and is not a placeholder
        if not openai_api_key or openai_api_key == "your-openai-api-key-here":
            return ChatResponse(
                session_id = req.session_id,
                response = "🔧 FinDeep AI is not configured. Please set up your OpenAI API key in the backend configuration."
            )

        # Create AI response using direct OpenAI integration (simplified approach)
        try:
            # Initialize OpenAI client with specific model and settings
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # Use efficient and cost-effective model
                api_key=openai_api_key,  # Pass the API key
                temperature=0.7  # Balance between creativity and consistency
            )
            
            # Define system prompt to make AI act as a financial analyst
            system_prompt = """You are FinDeep, an AI financial analyst assistant. You help users analyze financial data, create reports, and provide insights.

You can:
- Analyze financial documents and data
- Create financial reports and summaries
- Provide investment insights and recommendations
- Help with budgeting and forecasting
- Explain financial concepts and terms

Always provide clear, actionable financial insights. Be professional and helpful."""

            # Prepare messages for OpenAI API call
            messages = [
                {"role": "system", "content": system_prompt},  # System instruction
                {"role": "user", "content": req.message}  # User's question
            ]
            
            # Call OpenAI API to get AI response
            response = llm.invoke(messages)
            ai_response = response.content  # Extract the text content
            
            # Return successful response with session ID and AI message
            return ChatResponse(
                session_id = req.session_id,
                response = ai_response
            )
            
        except Exception as ai_error:
            # Handle OpenAI API specific errors (invalid key, rate limits, etc.)
            print(f"OpenAI API error: {ai_error}")
            return ChatResponse(
                session_id = req.session_id,
                response = f"🔧 AI Error: {str(ai_error)[:100]}... Please check your OpenAI API key and try again."
            )
            
    except Exception as e:
        # Handle any other unexpected errors
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code = 500, detail = str(e))