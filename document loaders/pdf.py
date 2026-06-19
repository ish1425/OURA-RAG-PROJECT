from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/MDP_notes.pdf")
docs = data.load()

print(len(docs))