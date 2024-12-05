# Import necessary modules and classes
from langchain_community.document_loaders import TextLoader  # For loading documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting documents into manageable chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For generating embeddings using Google GenAI
from langchain_community.vectorstores import FAISS  # For storing and retrieving document embeddings
from langchain_groq import ChatGroq  # For interacting with the Groq Chat LLM
from langchain_core.prompts import ChatPromptTemplate  # For creating chat-based prompts
from langchain.memory import ConversationSummaryMemory  # For managing conversational memory
from langchain_core.output_parsers import StrOutputParser  # For parsing LLM outputs
from dotenv import load_dotenv  # For loading environment variables

# Load environment variables (e.g., API keys) from a .env file
load_dotenv()

# Load the text document
loader = TextLoader("vikash.txt")
docs = loader.load()  # Load the document content

# Split the loaded documents into smaller chunks for processing
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
chunks = splitter.split_documents(docs)

# Initialize the embedding model for generating embeddings from text
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# Create a FAISS vector store to store embeddings of the document chunks
db = FAISS.from_documents(chunks, embedding_model)

# Configure a retriever to fetch relevant document chunks based on user queries
retriver = db.as_retriever()

# Initialize the LLM model for generating responses
llm = ChatGroq(model="llama-3.1-70b-versatile")

# Define a prompt template for the chat model
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who is supposed to summarize documents for the user. "
               "Please respond to the user's query based on the given context. "
               "If the answer is not found in the context, simply say 'Answer not found in given context'. "
               "Answer usual greeting and memory-based questions on past questions."),
    ("user", "The context is :{context}. The past conversation is : {history}. "
             "and the current question is {question}")
])

# Create a processing chain using the prompt, LLM, and output parser
chain = prompt | llm | StrOutputParser()

# Initialize memory for storing conversation summaries
memory = ConversationSummaryMemory(llm=llm)

# Main loop to handle user queries
while True:
    # Get the user's input query
    query = input("\nEnter the content you are looking for: ")

    # Load the conversation history from memory
    chat_history = memory.load_memory_variables({})['history']

    # Retrieve the relevant document chunk for the query
    context = retriver.invoke(query)[0].page_content

    # Use the chain to generate a response based on the context, history, and query
    result = chain.invoke({'context': context, 'history': chat_history, 'question': query})

    # Save the current query and its result into memory
    memory.save_context({"input": query}, {"outputs": result})

    # Display the response to the user
    print("\nResponse: " + result)
