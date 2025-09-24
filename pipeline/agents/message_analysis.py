from pipeline.constant.schema import GraphState, FinancialSchema
from pipeline.constant.prompt import MESSAGE_ANALYSIS_PROMPT

from dotenv import load_dotenv
load_dotenv()

import pytz
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from datetime import datetime

class MessageAnalysis(Runnable):
    def __init__(self, model_name: str, temperature: int = 0):
        self.__llm = ChatOpenAI(model = model_name, temperature = temperature).with_structured_output(FinancialSchema)
        self.__message_analysis_prompt = MESSAGE_ANALYSIS_PROMPT

    def invoke(self, state: GraphState, config = None):
        prompt = self.__message_analysis_prompt.format(
            user_message = state.user_message,
            current_time = str(datetime.now(pytz.utc))
        )
        response = self.__llm.invoke(prompt)
            
        state.start = response.start
        state.end = response.end
        state.value = response.value
        state.accn = response.accn
        state.fp = response.fp
        state.fy = response.fy
        state.form = response.form
        state.metric = response.metric
        state.CIK = response.CIK
        state.CompanyName = response.CompanyName
        return state