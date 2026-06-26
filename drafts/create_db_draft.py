#load pdf
#split into chunks
#create embeddings
#store in vector database (chroma)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader("document loaders/deeplearning.pdf") 
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = splitter.split_documents(docs)

embeddings_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents( #from_documents is a function that takes in the documents and embeddings and creates a vector database
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="chroma-db"
    )

