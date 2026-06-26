# load pdf
# split into chunks
# create embeddings
# store in vector database (chroma)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()


def create_db(pdf_paths):

    all_docs = [] #list to save all the pdfs that are uploaded

    # Load all uploaded PDFs
    for pdf_path in pdf_paths:

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        all_docs.extend(docs)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    # Create embeddings
    embeddings_model = OpenAIEmbeddings()

    # Store in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory="chroma-db"
    )

    return vectorstore

