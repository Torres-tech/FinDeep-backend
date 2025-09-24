from pipeline.constant.schema import GraphState
from pipeline.agents.message_analysis import MessageAnalysis
from pipeline.agents.message_systhesis import MessageSynthesis
from pipeline.agents.qdrant_retrieval import QdrantRetrieval

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

class GraphBuilder:
    def __init__(self, model_name:str, embedding_model:str):
        self.builder = StateGraph(GraphState)
        self.model_name = model_name
        self.embedding_model = embedding_model

    def build_graph(self):
        self.message_analysis = MessageAnalysis(model_name = self.model_name)
        self.qdrant_retrieval = QdrantRetrieval(embedding_model = self.embedding_model)
        self.message_synthesis = MessageSynthesis(model_name = self.model_name)

        self.builder.add_node("message_analysis", self.message_analysis)
        self.builder.add_node("qdrant_retrieval", self.qdrant_retrieval)
        self.builder.add_node("message_synthesis", self.message_synthesis)

        self.builder.add_edge(START, "message_analysis")
        self.builder.add_edge("message_analysis", "qdrant_retrieval")
        self.builder.add_edge("qdrant_retrieval", "message_synthesis")
        self.builder.add_edge("message_synthesis", END)

        return self.builder
    
class Graph:
    @staticmethod
    def compile(model_name:str, embedding_model:str):
        builder = GraphBuilder(model_name = model_name, embedding_model = embedding_model)
        memory = MemorySaver()
        return builder.build_graph().compile(checkpointer = memory)

def build_graph(model_name:str, embedding_model:str):
    graph = Graph.compile(model_name = model_name, embedding_model = embedding_model)
    return graph



