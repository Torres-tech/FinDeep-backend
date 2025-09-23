import os, uuid, pytz, torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

LIMITED_DATASET_SIZE = 10**6

class MiniLM_Embeddings:
    def __init__(self, model_name: str):
        csv_path = 'data_setup/sources/FinDeep_Query4.csv'
        self.__df = pd.read_csv(csv_path)
        print(self.__df.describe())

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = SentenceTransformer(model_name)
        self.__model.to(self.__device)
        print(f"Using device: {self.__device}")

        self.__collection_name = "FinDeep"
        self.__qdrant_client = QdrantClient(
            url = os.getenv("QDRANT_URL"),
            api_key = os.getenv("QDRANT_API_KEY")
        )

        self.__hashed_namespace = uuid.UUID(os.getenv("UUID_NAMESPACE"))
        self.__embedder = HuggingFaceEmbeddings(model_name = model_name)
    
    def __create_embeddings(self, save_path: str = 'data_setup/miniLM_embeddings.npy'):
        batch_size = 256
        embeddings_list = []

        print("Starting embedding generation...")
        for i in range(0, len(self.__df), batch_size):
            # Get a batch of texts from your 'Metric' column
            batch_texts = self.__df['Metric'][i:i+batch_size].tolist()

            # Use the .encode() method to generate embeddings for the batch
            # The model handles tokenization and processing internally.
            batch_embeddings = self.__model.encode(batch_texts, convert_to_tensor=True, device=self.__device)
            embeddings_list.append(batch_embeddings)

        print("Embedding generation complete.")
        # Concatenate all batches into a single tensor and move to CPU for saving.
        final_embeddings_gpu = torch.cat(embeddings_list)
        final_embeddings = final_embeddings_gpu.cpu().numpy()

        # Save the embeddings to a file for future use
        print(f"Embeddings generated and saved permanently to: {save_path}")
        np.save(save_path, final_embeddings)

        # Optionally, add them back to your DataFrame
        self.__df['miniLM_embedding'] = list(final_embeddings)

        print("Embeddings generated and saved!")
        print(self.__df.head())

    def __data_upload(self, save_path: str = 'data_setup/miniLM_embeddings.npy'):
        # Qdrant collection setup
        try:
            if self.__client.get_collection(self.__collection_name):
                print(f"Collection {self.__collection_name} already exists")
        except Exception as e:
            collection_config = models.VectorParams(
                size = 384, # vector dimension
                distance = models.Distance.COSINE
            )
            self.__client.create_collection(
                collection_name = self.__collection_name,
                vectors_config = collection_config
            )
            print(f"Created collection {self.__collection_name}")
            
        
        embeddings_list = np.load(save_path)
        print("Shape of embeddings_list:", embeddings_list.shape)
        points = []
        for i in range(embeddings_list.shape[0]):
            vector = embeddings_list[i]
            unique_id = str(datetime.now(pytz.utc))
            hashed_id = str(uuid.uuid5(self.__hashed_namespace, unique_id))
            points = PointStruct(
                    id = hashed_id,
                    vector = vector.tolist(),
                    payload = self.__df.iloc[i].to_dict()  # Store entire row
                )
            self.__client.upsert(
                collection_name = self.__collection_name,
                points=[points]
            )
        self.__client.close()

    def retrieve_query(self, query, top_k: int = 1):
        embedded_query = self.__embedder.embed_query(query)
        try:
            results = self.__qdrant_client.search(
                collection_name = self.__collection_name,
                query_vector = embedded_query,
                limit = top_k,
                with_payload = True,
                with_vectors = False
            )
            print(f'[INFO] From MiniLM_Embeddings.retrieve_query: Done search')
            return results
        except Exception as e:
            print(f"[ERROR] From MiniLM_Embeddings.retrieve_query: {str(e)}")
            return []

    def executor(self):
        # self.__create_embeddings()
        self.__data_upload()
    
if __name__ == "__main__":
    miniLM_embeddings = MiniLM_Embeddings("all-MiniLM-L6-v2")
    miniLM_embeddings.retrieve_query("2009-02-01	2010-01-31	4.08214E+11	0001193125-10-071652	2009	FY	10-K	Revenues	0000104169	Walmart")