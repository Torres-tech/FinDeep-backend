from pipeline.constant.schema import GraphState
from pipeline.agents.message_analysis import MessageAnalysis
from pipeline.agents.message_systhesis import MessageSynthesis
from pipeline.agents.qdrant_retrieval import QdrantRetrieval

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

class GraphBuilder:
    def __init__(
            self,
            embedding_model:str,
            model_name:str
        ):
        self.builder = StateGraph(GraphState)
        self.embedding_model = embedding_model
        self.model_name = model_name

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
    def compile(
        embedding_model:str,
        model_name:str
    ):
        builder = GraphBuilder(
            embedding_model = embedding_model,
            model_name = model_name, 
        )
        memory = MemorySaver()
        return builder.build_graph().compile(checkpointer = memory)

def build_graph(
        embedding_model:str,
        model_name:str, 
        save_graph:bool = False
    ):
    graph = Graph.compile(
        embedding_model = embedding_model,
        model_name = model_name
    )
    if save_graph:
        with open("pipeline/assets/chatbot_pipeline.png", "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
            print ("Image is saved to pipeline/assets/chatbot_pipeline.png")
    return graph