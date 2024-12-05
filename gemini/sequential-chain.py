from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# choose a model
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

# define as many prompts as desired
prompt1 = PromptTemplate.from_template("Who was the first PM of {country}")
prompt2 = PromptTemplate.from_template("What is the birth date of {PM}")

# chain each prompt with the llm in an seperate LLMChain mentioning its output_key
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key = 'PM')
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key = 'date')

# create an overall sequential chain combining the list of runnable LLMChains. Mention the list of input_variables and output_variables
chain = SequentialChain(chains=[chain1,chain2], input_variables = ['country'], output_variables = ["PM","date"])

# query the overall chain  print the final output_key
print(chain.invoke("India")['date'])