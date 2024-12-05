from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# choose a model
llm = ChatGroq(model="llama-3.1-70b-versatile")

# define as many prompts as desired
prompt1 = PromptTemplate.from_template("Who is the father of {person}?")
prompt2 = PromptTemplate.from_template("What is the nickname of {father}?")

# chain each prompt with the llm in an seperate LLMChain
chain1 = LLMChain(llm=llm,prompt=prompt1)
chain2 = LLMChain(llm=llm,prompt=prompt2)

# create an overall simple sequential chain combining the list of runnable LLMChains.
chain = SimpleSequentialChain(chains = [chain1,chain2], verbose=True)

# query the simple sequential chain & return the value assigned to ['output'] key
print(chain.invoke("Ranbir Kapoor")['output'])