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
from utils.helpers import handle_chat_mode, process_pdf_documents


def main():
    parser = argparse.ArgumentParser(
        prog='DocugentAi',
        description='DocugentAi CLI',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    # Subparser for chat mode
    chat_parser = subparsers.add_parser('chat', help='Enter chat mode')
    chat_parser.add_argument('--stream', action='store_true', help='Enable streaming responses')
    # Subparser for preparing the rag agent
    chat_parser = subparsers.add_parser('prepare', help='Inspect the contextual knowledge')

    # parse arguments
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    pm = PDFManger()
    pp = PDFProcessor()
    vs = ChromaDbService()
    model = ChatLocalOllamaMistral()
    
    if args.command == 'prepare':
        process_pdf_documents(pdf_manager=pm, pdf_processor=pp, vector_store=vs)
    if args.command == 'chat':
        handle_chat_mode(model=model, retriever=vs.get_retriever(k=2), stream=args.stream)

if __name__ == "__main__":
    main()