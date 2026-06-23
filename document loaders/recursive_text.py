from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
data = PyPDFLoader("document loaders/MDP_notes.pdf")
docs = data.load()

chunks = splitter.split_documents(docs)
print(chunks[0]) 