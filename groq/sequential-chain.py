from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# choose a model
llm = ChatGroq(model="llama-3.1-70b-versatile")

# define as many prompts as desired
prompt1 = PromptTemplate.from_template("Who is the father of {person}?")
prompt2 = PromptTemplate.from_template("What is the nickname of {father}?")

# chain each prompt with the llm in an seperate LLMChain mentioning its output_key
chain1 = LLMChain(llm=llm,prompt=prompt1, output_key = "father")
chain2 = LLMChain(llm=llm,prompt=prompt2, output_key = "nickname")

# create an overall chain combining the list of runnable LLMChains. Mention the list of input_variables and output_variables
chain = SequentialChain(chains=[chain1,chain2],input_variables=['person'], output_variables=['father','nickname'])

# query the overall chain  print the final output_key
print(chain.invoke("Ranbir Kapoor")['nickname'])