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
        Returns list a list of id (str) corresponding to each passed lanchain document
        """
        from utils.helpers import generate_unique_id # to avoid circular imports
        document_ids = []
        for i, doc in enumerate(documents, start=1):
            filename = doc.metadata.get('filename', 'unknown')
            # unique key for each document chunk
            hash_key = f"{filename}-doc-{i}"
            doc_id = generate_unique_id(hash_key)
            document_ids.append(doc_id)

        return document_ids