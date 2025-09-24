from pipeline.constant.schema import GraphState
from pipeline.constant.prompt import MESSAGE_SYNTHESIS_PROMPT

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage

class MessageSynthesis(Runnable):
    def __init__(self, model_name:str, temperature:int = 0):
        self.__llm = ChatOpenAI(model = model_name, temperature = temperature)
        self.__message_synthesis_prompt = MESSAGE_SYNTHESIS_PROMPT

    def invoke(self, state:GraphState, config = None):
        prompt = self.__message_synthesis_prompt.format(
            user_message = state.user_message,
            data = state.data_metadata
        )
        response = self.__llm.invoke(prompt)
        state.chat_history.append(HumanMessage(content = state.user_message))
        state.chat_history.append(AIMessage(content = response.content))
        return state