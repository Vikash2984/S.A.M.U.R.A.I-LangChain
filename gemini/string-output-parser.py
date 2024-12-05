from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv 

# load environment variables
load_dotenv()

# choose a model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# define a string output parser instance 
output_parser = StrOutputParser()

# chain the output parser in your runnable sequence
chain = llm | output_parser

# directly print response as a string
print(chain.invoke("which bollywood actor is nicknamed Chintu?"))