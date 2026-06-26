from dotenv import load_dotenv  
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma #to retireve the data from the vector database created alr
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma(
    persist_directory = "chroma-db",
    embedding_function = embedding_model
) #loaded the vector database

#now we create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k":4, "fetch_k":10, "lambda_mult": 0.5}
    ) #k is the number of similar documents we want to retrieve. In this case, we are retrieving the top 2 similar documents based on the query.
#fetch_k is the number of documents we want to fetch from the vector database. In this case, we are fetching the top 10 similar documents based on the query, and then from these 10 documents, we retrieve top 4 similar documents based on the query. 
#lambda_mult is the weight given to the relevance/diversity score of the documents. In this case, we are giving equal weight to the relevance score and the diversity score of the documents.

llm = ChatMistralAI(
    model="mistral-small-2506"
)

#prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """you are a helpful assistant that summarizes.
     Use ONLY the provided context to answer the question.
     If the context is not sufficient to answer the question, say "I could not find the information to answer the question."
     """),
    ("human", """Context: {context} 
     Question: {question}""")
]) #context from the retiever and question from the user. 

print("Welcome to the RAG system. Type '0' to quit.")

while True:
    query = input("You: ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context": context, "question": query})

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")

