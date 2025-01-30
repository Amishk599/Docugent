from uuid import uuid4
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, texts: List[str], metadatas: List[dict]) -> List[Document]:
        """
        Breaks the PDF text into chunk of specified size
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " "],
        )
        return text_splitter.create_documents(texts=texts, metadatas=metadatas)
    
    def generate_ids_for_documents(self, documents: List[Document]) -> List[str]:
        """
        Returns list a list of uuids (str) corresponding to each passed lanchain document
        """
        return [str(uuid4()) for _ in range(len(documents))]