import os
import sys
import argparse

# load environment variables
from dotenv import load_dotenv
load_dotenv()

from services.pdf_manager import PDFManger
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
    
    model_name = os.getenv('OLLAMA_MODEL_NAME')
    document_directory = os.getenv('DOCUMENTS_PATH')

    llm = ChatLocalOllamaMistral(model_name=model_name)
    pdf_manager = PDFManger(docs_dir=document_directory)
    
    if args.command == 'chat':
        handle_chat_mode(args, llm)

if __name__ == "__main__":
    main()