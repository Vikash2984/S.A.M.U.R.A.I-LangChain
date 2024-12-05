from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# choose the desired model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# query the language model
result = llm.invoke("Who is the Prime Minister of India?")

# print response as desired
print("Response : ",result.content)