from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# Initialize the language model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Initialize memory
memory = ConversationSummaryMemory(llm=llm)

# Create a conversation chain
chain = ConversationChain(llm=llm, memory=memory)

while True : 
    query = input("\nEnter a prompt : ")

    # query the conversation
    result = chain.invoke(query)

    # print the response
    print("\nResponse : "+result['response']) 
    