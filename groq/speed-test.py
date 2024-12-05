from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGCHAIN_TRACING"]="true"

gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llama = ChatGroq(model="llama-3.1-70b-versatile")

while True:
    query = input("\nEnter a prompt : ")
    print("\nGemnin : "+gemini.invoke(query).content)
    print("\nLlama : "+llama.invoke(query).content)