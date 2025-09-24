from pipeline.constant.schema import GraphState
from pipeline.constant.prompt import QDRANT_RETRIEVAL_PROMPT

from dotenv import load_dotenv
load_dotenv()

import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain_core.runnables import Runnable

class QdrantRetrieval(Runnable):
    def __init__(self, embedding_model):
        self.__collection_name = "FinDeep"
        self.__qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.__model = SentenceTransformer(embedding_model)
        self.__qdrant_retrieval_prompt = QDRANT_RETRIEVAL_PROMPT
    
    def retrieve_query(self, query, top_k: int = 1):
        embedded_query = self.__model.encode(query)
        try:
            results = self.__qdrant_client.search(
                collection_name = self.__collection_name,
                query_vector = embedded_query,
                limit = top_k,
                with_payload = True,
                with_vectors = False
            )
            return results
        except Exception as e:
            print(f"[ERROR] From QdrantRetrieval: {e}")
            exit(1)
    
    def invoke(self, state: GraphState, config = None):
        query = self.__qdrant_retrieval_prompt.format(
            start = state.start,
            end = state.end,
            value = state.value,
            accn = state.accn,
            fp = state.fp,
            fy = state.fy,
            form = state.form,
            metric = state.metric,
            CIK = state.CIK,
            CompanyName = state.CompanyName
        )
        response = self.retrieve_query(query)
        state.data_metadata = response[0].payload["metadata"]
        state.data_position = str(response[0].payload["position"] + 2)
        return state