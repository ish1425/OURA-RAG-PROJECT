from langchain_community.document_loaders import TextLoader

data = TextLoader("document loaders/notes.txt")
docs = data.load()

print(docs[0].page_content) #the doc is stored in the form of list with metadata and pagecontent.
#print(docs) #this will give list with metadata and pagecontent
