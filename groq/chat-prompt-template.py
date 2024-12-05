from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# choose a model
llm = ChatGroq(model="llama-3.1-70b-versatile")

# define the system prompt to govern the behaviour of the llm and user prompt to define the flow of user interaction
prompt = ChatPromptTemplate.from_messages([("system","You are professinal standup comic like Samay raina roast the user but keep the responses brief"),("user","The current quesion is : {question}")])

# pair the prompt with LLM to from a Runnable Sequence
chain = prompt | llm

while True:
    query = input("\nEnter a prompt : ")
    # query the Runnable & print the desired response
    print("\nRespnse : "+chain.invoke(query).content)