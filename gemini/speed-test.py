from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# load enviromet variabls
load_dotenv()

# enable LangChain Tracing
os.environ["LANGCHAIN_TRACING"]="true"

# Choose a model for Gemini
gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Choose a model for Llama
llama = ChatGroq(model="llama-3.1-70b-versatile")

# Run Gemini and llama simultaneously
while True:
    query = input("\nEnter a prompt : ")
    print("\nGemnin : "+gemini.invoke(query).content)
    print("\nLlama : "+llama.invoke(query).content)
