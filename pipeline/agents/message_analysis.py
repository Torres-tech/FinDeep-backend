from pipeline.constant.schema import GraphState, FinancialSchema
from pipeline.constant.prompt import MESSAGE_ANALYSIS_PROMPT

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable

class MessageAnalysis(Runnable):
    def __init__(self, model_name: str, temperature:int = 0):
        self.__llm = ChatOpenAI(model = model_name, temperature = temperature).with_structured_output(FinancialSchema)
        self.__message_analysis_prompt = MESSAGE_ANALYSIS_PROMPT

    def invoke(self, state: GraphState, config = None):
        prompt = [
            SystemMessage(content = self.__message_analysis_prompt),
            HumanMessage(content = f"USER MESSAGE: {state.user_message}")
        ]
        response = self.__llm.invoke(prompt)
            
        state.start = response.start
        state.end = response.end
        state.value = response.value
        state.accn = response.accn
        state.fp = response.fp
        state.fy = response.fy
        state.form = response.form
        state.metric = response.metric
        state.cik = response.cik
        state.companyname = response.companyname
        return state