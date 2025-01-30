from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate

def handle_chat_mode(model, retriever):
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
                # response = model.invoke([HumanMessage(content=question)])
                # trigger the rag pipeline and generate response
                response = rag_pipeline(
                    model=model, 
                    retriever=retriever, 
                    question=question,
                )
                print(f"DocugentAi: {response.content}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting chat mode")

def rag_pipeline(model, retriever, question):
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

    # generate the response using the chain
    response = chain.invoke({"context":context, "question":question})

    return response

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