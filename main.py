import os
import sys
import argparse

# load environment variables
from dotenv import load_dotenv
load_dotenv()

from services.pdf_manager import PDFManger
from services.pdf_processor import PDFProcessor
from services.chroma_db import ChromaDbService
from services.ollama_custom import ChatLocalOllamaMistral
from langchain.schema import HumanMessage
from utils.helpers import handle_chat_mode


def main():
    parser = argparse.ArgumentParser(
        prog='DocugentAi',
        description='DocugentAi CLI',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    # Subparser for chat mode
    chat_parser = subparsers.add_parser('chat', help='Enter chat mode')
    # parse arguments
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    llm = ChatLocalOllamaMistral()
    pdf_manager = PDFManger()
    pdf_processor = PDFProcessor()
    vector_store = ChromaDbService()

    # count of PDF documents present.
    docs = pdf_manager.list_all_docs()
    # ensure vector store is empty
    vecs = vector_store.get_vectors_count()
    print(f"PDFs Count[{len(docs)}]\n")

    for doc in docs:
        # read PDF content from all pages
        text = pdf_manager.read_pdf(doc)
        print(f"PDF Content:\n{text}\n\n")
        # split the PDF text and form langchain documents and give each an id
        documents = pdf_processor.chunk_text(texts=[text], metadatas=[{"filename": doc}])
        print(f"After chunking:\n{documents}\n\n")
        document_ids = pdf_processor.generate_ids_for_documents(documents)
        # convert documents to embeddings add add to vector store
        vector_store.add_document(documents=documents, ids=document_ids)

    vecs = vector_store.get_vectors_count()
    print(f"Vectors[{vecs}]\n\n")
    
    if args.command == 'chat':
        handle_chat_mode(args, llm)

if __name__ == "__main__":
    main()