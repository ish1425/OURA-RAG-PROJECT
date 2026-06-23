from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter
splitter = CharacterTextSplitter(separator="",chunk_size=10, chunk_overlap=1)

#data = TextLoader("document loaders/notes.txt") #example of simple text file loading
data = TextLoader("document loaders/notes1.txt") #example of character splitter
docs = data.load()

chunks = splitter.split_documents(docs)

#print(docs[0].page_content) #the doc is stored in the form of list with metadata and pagecontent.
#print(docs) #this will give list with metadata and pagecontent
#print(chunks) #this will give the splitted chunks of the document.

for i in chunks:
    print(i.page_content) 
    print()
    print()
    print()