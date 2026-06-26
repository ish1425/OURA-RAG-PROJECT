import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="Oura",
    page_icon="📚",
    layout="wide"
)

# ----------------------------
# CSS
# ----------------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1000px;
}

.main-title{
    font-size:42px;
    font-weight:700;
    text-align:center;
}

.sub-title{
    text-align:center;
    color:gray;
    margin-bottom:25px;
}

.upload-box{
    border:1px solid #ddd;
    padding:15px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------
# Session State
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False


# ----------------------------
# Header
# ----------------------------

st.markdown(
    "<div class='main-title'>📚 Oura</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Your AI Book Assistant</div>",
    unsafe_allow_html=True
)


# ----------------------------
# Sidebar
# ----------------------------

with st.sidebar:

    st.header("Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type="pdf"
    )

    if st.button("Create Knowledge Base"):

        if uploaded_file is None:
            st.warning("Please upload a PDF first.")

        else:

            os.makedirs("document loaders", exist_ok=True)

            pdf_path = os.path.join(
                "document loaders",
                uploaded_file.name
            )

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if os.path.exists("chroma-db"):
                shutil.rmtree("chroma-db")

            with st.spinner("Creating embeddings..."):

                loader = PyPDFLoader(pdf_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(docs)

                embeddings = OpenAIEmbeddings()

                Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma-db"
                )

            st.success("Knowledge Base Created!")

            st.session_state.db_ready = True

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ----------------------------
# Load Vector DB
# ----------------------------

if st.session_state.db_ready or os.path.exists("chroma-db"):

    embedding_model = OpenAIEmbeddings()

    vectorstore = Chroma(
        persist_directory="chroma-db",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    prompt = ChatPromptTemplate.from_messages([

        ("system",
        """You are a helpful assistant that summarizes.

Use ONLY the provided context.

If the context is insufficient say:

"I could not find the information to answer the question."
"""),

        ("human",
        """Context:

{context}

Question:

{question}
""")
    ])

# ----------------------------
# Chat History
# ----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ----------------------------
# Chat Input
# ----------------------------

user_question = st.chat_input("Ask something about your uploaded book...")

if user_question:

    if not os.path.exists("chroma-db"):

        st.warning("Please upload a PDF and create the knowledge base first.")

    else:

        st.session_state.messages.append(
            {
                "role":"user",
                "content":user_question
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                docs = retriever.invoke(user_question)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                final_prompt = prompt.invoke({
                    "context":context,
                    "question":user_question
                })

                response = llm.invoke(final_prompt)

                st.markdown(response.content)

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":response.content
            }
        )