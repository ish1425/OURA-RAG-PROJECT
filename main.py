from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate #we use chatprompttemplate so we can give roles
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()


#data = TextLoader("document loaders/notes.txt") #this is how you import text file
data = PyPDFLoader("document loaders/deeplearning.pdf") 
docs = data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = splitter.split_documents(docs)


template = ChatPromptTemplate.from_messages([ #from_messages is used to give roles to the prompt. we can give system and human roles.
    ("system", "you are a helpful assistant that summarizes the text"),
    ("human", "{data}")
])

model = ChatMistralAI(
    model="mistral-small-2506"
)

prompt = template.format_messages(data=docs[0].page_content) #in the context of pdf, the [0] is used to get the first page of the pdf.

result = model.invoke(prompt)
print(result.content)