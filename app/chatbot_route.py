from app.request_schema import ChatRequest, ChatResponse

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import os

router = APIRouter()

@router.post("/chat", response_model = ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key or openai_api_key == "your-openai-api-key-here":
            return ChatResponse(
                session_id = req.session_id,
                response = "🔧 FinDeep AI is not configured. Please set up your OpenAI API key in the backend configuration."
            )

        # Create a simple AI response using OpenAI directly
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=openai_api_key,
                temperature=0.7
            )
            
            # Create a financial analysis prompt
            system_prompt = """You are FinDeep, an AI financial analyst assistant. You help users analyze financial data, create reports, and provide insights.

You can:
- Analyze financial documents and data
- Create financial reports and summaries
- Provide investment insights and recommendations
- Help with budgeting and forecasting
- Explain financial concepts and terms

Always provide clear, actionable financial insights. Be professional and helpful."""

            # Get AI response
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message}
            ]
            
            response = llm.invoke(messages)
            ai_response = response.content
            
            return ChatResponse(
                session_id = req.session_id,
                response = ai_response
            )
            
        except Exception as ai_error:
            print(f"OpenAI API error: {ai_error}")
            return ChatResponse(
                session_id = req.session_id,
                response = f"🔧 AI Error: {str(ai_error)[:100]}... Please check your OpenAI API key and try again."
            )
            
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code = 500, detail = str(e))