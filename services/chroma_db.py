import chromadb
from typing import List
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

class ChromaDbService:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        # embedding_dimension is 768
        self.embedding_dimension = len(self.embedding_model.embed_query("hello world"))
        # TODO - currently it is an in-memory store. Later change it to a persistent store
        self.client = Chroma(
            client=chromadb.EphemeralClient(),
            collection_name="docugent_vectore_store",
            embedding_function=self.embedding_model
        )

    def add_document(self, documents: List[Document], ids):
        """
        Add document to ChromaDb
        """
        self.client.add_documents(documents=documents, ids=ids)

    def get_vectors_count(self):
        """
        Returns the number of vectores present in ChromaDb
        """
        self.client.as_retriever
        return len(self.client.get()['ids'])
    
    def get_retriever(self, search_type: str = "mmr", k: int = 4) -> VectorStoreRetriever:
        """
        Return VectorStoreRetriever object
        """
        return self.client.as_retriever(
            search_type=search_type, 
            search_kwargs={"k": k},
        )