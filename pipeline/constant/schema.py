from typing import Annotated, List, Dict, Any, Optional
from pydantic import BaseModel
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class GraphState(BaseModel):
    chat_history: Annotated[List[AnyMessage], add_messages]
    user_message: str = ""
    retrieved_data: Optional[List[Any]] = None
    # Financial Schema
    start: Optional[str] = ""
    end: Optional[str] = ""
    value: Optional[str] = ""
    accn: Optional[str] = ""
    fp: Optional[str] = ""
    fy: Optional[str] = ""
    form: Optional[str] = ""
    metric: Optional[str] = ""
    cik: Optional[str] = ""
    companyname: Optional[str] = ""

class FinancialSchema(BaseModel):
    start: str
    end: str
    value: str
    accn: str
    fp: str
    fy: str
    form: str
    metric: str
    cik: str
    companyname: str