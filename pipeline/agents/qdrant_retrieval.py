from pipeline.constant.schema import GraphState
from pipeline.constant.prompt import QDRANT_RETRIEVAL_PROMPT

from dotenv import load_dotenv
load_dotenv()

import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.runnables import Runnable

class QdrantRetrieval(Runnable):
    def __init__(self, embedding_model):
        self.__collection_name = "FinDeep"
        self.__collection_keys = [
            "start", 
            "end", 
            "value", 
            "accn", 
            "fp", 
            "fy", 
            "form", 
            "metric", 
            "CIK", 
            "CompanyName"
        ]
        self.__qdrant_client = QdrantClient(
            url = os.getenv("QDRANT_URL"),
            api_key = os.getenv("QDRANT_API_KEY")
        )
        self.__model = SentenceTransformer(embedding_model)
        self.__qdrant_retrieval_prompt = QDRANT_RETRIEVAL_PROMPT
    
    def __retrieve_query(self, query, filter_dict: dict, top_k: int = 1000):
        embedded_query = self.__model.encode(query)
        must_conditions = []
        for key in self.__collection_keys:
            if filter_dict[key] != "":
                must_conditions.append(
                    models.FieldCondition(
                        key = f"metadata.{key}",
                        match = models.MatchValue(value = filter_dict[key])
                    )
                )
        
        query_filter = models.Filter(must = must_conditions) if must_conditions else None
        # results = self.__qdrant_client.scroll(
        #     collection_name = self.__collection_name,
        #     scroll_filter = query_filter,
        #     with_payload = True,
        #     with_vectors = False,
        #     limit = top_k
        # )
        try:
            results = self.__qdrant_client.search(
                collection_name = self.__collection_name,
                query_vector = embedded_query,
                query_filter = query_filter,
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
            CIK = state.cik,
            CompanyName = state.companyname
        )
        
        def safe_convert(value):
            try:
                flag = int(value)
                return True
            except:
                return False
        
        response = self.__retrieve_query(
            query,
            dict (
                start = "",
                end = "",
                value = int(state.value) if safe_convert(state.value) else "",
                accn = state.accn,
                fp = state.fp,
                fy = int(state.fy) if safe_convert(state.fy) else "",
                form = state.form,
                metric = state.metric,
                CIK = int(state.cik) if safe_convert(state.cik) else "",
                CompanyName = state.companyname
            )
        )
        state.retrieved_data = response
        return state