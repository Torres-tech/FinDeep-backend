from app.request_schema import ChatRequest, ChatResponse

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()

@router.post("/chat", response_model = ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        init_state = {"user_message": req.message}

        try:
            result = router.graph.invoke(
                input = init_state,
                config = {"configurable": {"thread_id": req.session_id}}
            )
        except Exception as e:
            print(e)

        ai_msg = result["chat_history"][-1]
        if isinstance(ai_msg, AIMessage):
            reply_text = ai_msg.content
        else:
            reply_text = "Sorry, I didn't understand that."

        return ChatResponse(
            session_id = req.session_id,
            response = reply_text
        )
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))