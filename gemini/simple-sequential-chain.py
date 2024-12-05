from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# choose a model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# define as many prompts as desired
prompt1 = PromptTemplate.from_template("Who was the eldest member of {group}?")
prompt2 = PromptTemplate.from_template("What is the most heard song of {artist}?")

# chain each prompt with the llm in an seperate LLMChain
chain1 = LLMChain(llm=llm,prompt=prompt1)
chain2 = LLMChain(llm=llm,prompt=prompt2)

# create an overall simple sequential chain combining the list of runnable LLMChains.
chain = SimpleSequentialChain(chains=[chain1,chain2],verbose=True)

# query the simple sequential chain & return the value assigned to ['output'] key
print(chain.invoke("One Direction")['output'])