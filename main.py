from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
#from langchain_community.document_loaders import TextLoader #for importing .txt file
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate #we use chatprompttemplate so we can give roles

load_dotenv()


#data = TextLoader("document loaders/notes.txt") #this is how you import text file
data = PyPDFLoader("document loaders/MDP_notes.pdf") 
docs = data.load()

template = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant that summarizes the text"),
    ("human", "{data}")
])

model = ChatMistralAI(
    model="mistral-small-2506"
)

prompt = template.format_messages(data=docs[0].page_content) #in the context of pdf, the [0] is used to get the first page of the pdf.

result = model.invoke(prompt)
print(result.content)