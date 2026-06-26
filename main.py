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
    ("system", """You are Oura, an AI document assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not make up information.
- If the answer is not present in the context, say:
"I could not find the information to answer the question."

- Give clear and concise answers.
- Use markdown formatting when helpful.
- Use previous conversation context when answering follow-up questions.

     """),
    ("human", """Conversation history: {history}
     Context: {context} 
     Question: {question}""")
]) #context from the retiever and question from the user. 
#convo history is the previous conversation between the user and the AI. This is used to answer follow-up questions.

chat_history = [] #list to save the previous conversation between the user and the AI. This is used to answer follow-up questions.

print("Welcome to the RAG system. Type '0' to quit.")

while True:
    query = input("You: ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    history = "\n".join(
        [
            f"User: {msg['user']}\nAI: {msg['ai']}"
            for msg in chat_history
        ]
    )

    final_prompt = prompt.invoke({"context": context, "question": query, "history": history})

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")

     # display sources

    print("\n Sources:")


    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get(
            "source",
            "unknown"
        )


        page = doc.metadata.get(
            "page"
        )


        if page is not None:
            page = page + 1


        print(
            f"{i}. {source} - Page {page}"
        )



    print("\n")


    # save memory

    chat_history.append(
        {
            "user": query,
            "ai": response.content
        }
    )

