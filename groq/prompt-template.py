from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# choose a model
llm = ChatGroq(model="llama-3.1-70b-versatile")

# define the prompt with as many place holders as desired
prompt = PromptTemplate.from_template("The capital of {place} is?")

# pair the prompt with LLM to from a Runnable Sequence
chain = prompt | llm

# query the Runnable & print the desired response
print("Response : "+chain.invoke("India").content)