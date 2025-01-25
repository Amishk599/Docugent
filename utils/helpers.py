from langchain.schema import HumanMessage

def handle_chat_mode(args, llm):
    """
    Handle interactive chat mode
    """
    print("Entering chat mode. Type 'exit' or 'quit' to leave.\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting chat mode\n")
                break
            elif not user_input:
                continue
            # generate response
            if llm:
                response = llm.invoke([HumanMessage(content=user_input)])
                print(f"DocugentAi: {response.content}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting chat mode") 