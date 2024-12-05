from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# choose a model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# define the prompt with as many place holders as desired
prompt = PromptTemplate.from_template("You are a helpful assistant who has been assigned the work to provide the user with the name of capital cities of places. The quesion is : The capital of {city} is? Don't reply any questions other than the capital of a given place.")

# form a runnable sequence 
chain = prompt | llm

# query the Runnable & print the response 
print("\n",chain.invoke("China").content)