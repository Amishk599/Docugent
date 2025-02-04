import hashlib
from services.pdf_manager import PDFManger
from services.pdf_processor import PDFProcessor
from services.chroma_db import ChromaDbService
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate

def handle_chat_mode(model, retriever, stream=False):
    """
    Handle interactive chat mode
    """
    print("Entering chat mode. Type '/exit' or '/quit' to leave.\n")
    try:
        while True:
            question = input("You: ").strip()
            if question.lower() in ['/exit', '/quit']:
                print("Exiting chat mode\n")
                break
            elif not question:
                continue
            
            if model and retriever:
                print("DocugentAi: ", end="", flush=True)
                # print the response (streaming or non-streaming)
                if stream:
                    for chunk in rag_pipeline(model, retriever, question, stream=True):
                        print(chunk.content, end="", flush=True)
                    
                else:
                    response = rag_pipeline(model, retriever, question)
                    print(response.content, end="", flush=True)
                
                print("\n")  
    except (KeyboardInterrupt, EOFError):
        print("\nExiting chat mode")

def rag_pipeline(model, retriever, question, stream=False):
    """
    Main pipeline where based on question, relevant documents are 
    retrived and model uses it for context to answer the question.
    """
    # get the predefined prompt template for docugent
    template = get_prompt_template()

    # retrieve relevant documents from ChromaDb
    relevant_documents = retriever.invoke(question)

    # extract retrived documents content
    context = "\n\n".join([doc.page_content for doc in relevant_documents])

    # define the prompt structure for chain
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # define chain to process the response
    chain = prompt | model

    if stream:
        return chain.stream({"context":context, "question":question})
    else:
        return chain.invoke({"context":context, "question":question})

def process_pdf_documents(
        pdf_manager: PDFManger, 
        pdf_processor: PDFProcessor, 
        vector_store: ChromaDbService
    ):
    print("Starting to process all PDF documents...")
    docs = pdf_manager.list_all_docs()
    vecs_count_at_start = vector_store.get_vectors_count()

    for doc in docs:
        hash_key = f"{doc}-doc-1"
        doc_id = generate_unique_id(hash_key)
        if vector_store.document_exists(doc_id=doc_id):
            print(f"Skipping {doc}, vectors already exist in store for this file")
            continue
        # read PDF content from all pages
        text = pdf_manager.read_pdf(doc)
        # split the PDF text and form langchain documents and give each an id
        documents = pdf_processor.chunk_text(texts=[text], metadatas=[{"filename": doc}])
        document_ids = pdf_processor.generate_ids_for_documents(documents)
        # convert documents to embeddings add add to vector store
        vector_store.add_document(documents=documents, ids=document_ids)
        print(f"Processed {doc}, vectors created and added to store")

    vecs_count_at_end = vector_store.get_vectors_count()
    print(f"\nFound {len(docs)} PDFs\nvectors at start: {vecs_count_at_start}\nvectors at end: {vecs_count_at_end}")

def generate_unique_id(hash_key: str):
    """
    generates a unique id based a hash key
    """
    return hashlib.md5(hash_key.encode()).hexdigest()

def get_prompt_template():
    """
    Returns the predefined prompt template for Docugent
    """
    template = """Use the following pieces of context to answer the question. If you don't know the answer, say so.
    Context:
    {context}
    Question:
    {question}
    Answer:"""
    return template