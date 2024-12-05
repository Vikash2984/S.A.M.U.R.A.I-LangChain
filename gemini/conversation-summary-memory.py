from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv

# laod environment variables
load_dotenv()

# choose a model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# define a memory instance
memory = ConversationSummaryMemory(llm=llm)

while True:
    query = input("\nEnter a prompt : ")

    # define chat history variable to store past conversation
    chat_history = memory.load_memory_variables({})['history']

    # query the LLM
    result = llm.invoke(query)
    print("\nResponse : "+result.content)

    # update memory with the current query and response
    memory.save_context({"input":query},{"outputs":result.content})

    # print chat history
    print("\n"+memory.buffer)