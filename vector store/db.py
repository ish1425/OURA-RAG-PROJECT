from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document

docs = [
    Document(page_content="Q1 revenue increased by 15 due to strong iPhone sales.",metadata={"source": "apple_q1_report.pdf", "page": 3}),
    Document(page_content="Services revenue grew by 12 due to subscriptions.", metadata={"source": "nvidia_report.pdf", "page": 5}),
    Document(page_content="Greater China recorded the highest regional growth this quarter.", metadata={"source": "micro_report.pdf", "page": 8})
]

embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "chroma-db"
    )

result = vectorstore.similarity_search("What was the Q1 revenue increase due to?", k=2) #k stands for the number of similar documents you want to retrieve. In this case, we are retrieving the top 2 similar documents based on the query.

for r in result:
    print (r)

