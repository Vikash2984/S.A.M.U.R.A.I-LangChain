# Import necessary libraries and modules
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the text document using the TextLoadedocument_loaders.ipynbr
loader = TextLoader("langchain.txt")
docs = loader.load()  # Load documents from the specified file

# Split the documents into manageable chunks
# Chunk size and overlap are defined to ensure meaningful segmentation
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=500, chunk_size=500)
chunks = text_splitter.split_documents(docs)

# Initialize the embedding model for vector representation of text
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create a FAISS vector store from the document chunks and embedding model
db = FAISS.from_documents(chunks, embedding_model)

# Set up a retriever to fetch relevant context from the vector store
retriver = db.as_retriever()

# Take user input as a query
query = input("Enter a prompt: ")

# Use the retriever to get relevant context for the query
context = retriver.invoke(query)

# Initialize the LLM (Large Language Model) for generating responses
llm = ChatGroq(model="llama-3.1-70b-versatile")

# Define a prompt template to structure the AI's response
prompt = PromptTemplate.from_template(
    "You are a helpful chatbot who is supposed to answer questions based on context provided by the user, which is: {context}. "
    "If the answer to the user's question is not found in the context, say 'Answer not found in provided context'. "
    "The question of the user is {question}."
)

# Combine the prompt and the LLM into a chain for execution
chain = prompt | llm

# Invoke the chain with the context and user's question to get a response
result = chain.invoke({"context": context, "question": query})

# Print the response from the AI
print("\nResponse: " + result.content)
