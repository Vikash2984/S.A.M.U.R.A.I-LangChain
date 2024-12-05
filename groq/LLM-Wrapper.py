from langchain_groq import ChatGroq
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# choose the desired model
llm = ChatGroq(model="llama-3.1-70b-versatile")

# query the language model
result = llm.invoke("Who is the Prime Minister of India?")

# print response as desired
print("\nResponse : ",result.content)