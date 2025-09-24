import os, uuid, pytz, torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

class MiniLM_Embeddings:
    def __init__(
            self, 
            embedding_model: str, 
            xlsx_path: str,
            csv_path: str,
            save_path: str
        ):
        self.__csv_path = csv_path
        self.__xlsx_path = xlsx_path
        self.__save_path = save_path
        self.__hashed_namespace = uuid.UUID(os.getenv("UUID_NAMESPACE"))

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = SentenceTransformer(embedding_model)
        self.__model.to(self.__device)
        print(f"Using device: {self.__device}")

        self.__collection_name = "FinDeep"
        self.__qdrant_client = QdrantClient(
            url = os.getenv("QDRANT_URL"),
            api_key = os.getenv("QDRANT_API_KEY")
        )
    
    def __create_embeddings(self):
        def create_prompt_text(row):
            return (
                f"start:{row['start']},"
                f"end:{row['end']},"
                f"value:{row['value']},"
                f"accn:{row['accn']},"
                f"fp:{row['fp']},"
                f"fy:{row['fy']},"
                f"form:{row['form']},"
                f"metric:{row['metric']},"
                f"CIK:{row['CIK']},"
                f"CompanyName:{row['CompanyName']}"
            )
        
        df = pd.read_excel(self.__xlsx_path)
        df['prompt_text'] = df.apply(create_prompt_text, axis=1)
        documents = df['prompt_text'].tolist()
        embeddings = self.__model.encode(documents)
        np.save(self.__save_path, embeddings)

    def __data_upload(self):
        # Qdrant collection setup
        try:
            if self.__qdrant_client.get_collection(self.__collection_name):
                print(f"Collection {self.__collection_name} already exists")
        except Exception as e:
            collection_config = models.VectorParams(
                size = 384, # vector dimension
                distance = models.Distance.COSINE
            )
            self.__qdrant_client.create_collection(
                collection_name = self.__collection_name,
                vectors_config = collection_config
            )
            print(f"Created collection {self.__collection_name}")
            
        # Start uploading
        embeddings_list = np.load(self.__save_path)
        print("Shape of embeddings_list:", embeddings_list.shape)
        points = []
        df = pd.read_csv(self.__csv_path)
        for i in range(embeddings_list.shape[0]):
            vector = embeddings_list[i]
            unique_id = str(datetime.now(pytz.utc))
            hashed_id = str(uuid.uuid5(self.__hashed_namespace, unique_id))
            points = PointStruct(
                    id = hashed_id,
                    vector = vector.tolist(),
                    payload = {
                        "metadata": df.iloc[i].to_dict(),
                        "position": i
                    }  # Store entire row
                )
            self.__qdrant_client.upsert(
                collection_name = self.__collection_name,
                points=[points]
            )
        self.__qdrant_client.close()

    def executor(self):
        # self.__create_embeddings()
        self.__data_upload()

if __name__ == "__main__":
    Nhi = MiniLM_Embeddings(
        'sentence-transformers/all-MiniLM-L6-v2',
        'data_setup/sources/FinDeep_data (cleaned).xlsx',
        'data_setup/sources/FinDeep_data (cleaned).csv',
        'data_setup/sources/financial_embeddings.npy'
    )
    Nhi.executor()