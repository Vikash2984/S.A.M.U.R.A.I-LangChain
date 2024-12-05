from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# choose a model
llm= ChatGroq(model="llama-3.1-70b-versatile")

# define a string output parser instance 
output_parser = StrOutputParser()

# chain the output parser in your runnable sequence
chain = llm | output_parser

# directly print response as a string
print("\n"+chain.invoke("which bollywood celeb is nicknamed Duggu?"))