from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# load enviorment variables
load_dotenv()

# choose a model
llm = ChatGroq(model="llama-3.1-70b-versatile")

# define a memory instance
memory = ConversationBufferMemory(llm=llm)

while True:
    query = input("\nEnter a prompt : ")

    # define chat history variable to store past conversation
    chat_history = memory.load_memory_variables({})['history']

    # query the LLM
    result = llm.invoke(query)
    print ("\nResponse : ",result.content)

    # update memory with the current query and response
    memory.save_context({"input":query},{"outputs":result.content})

    # print chat history
    print("\n"+memory.buffer)