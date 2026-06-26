import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from create_db import create_db


load_dotenv()


# -----------------------
# Page config
# -----------------------

st.set_page_config(
    page_title="Oura",
    page_icon="📚",
    layout="wide"
)


# -----------------------
# Styling
# -----------------------

st.markdown(
"""
<style>

.main-title{
    text-align:center;
    font-size:45px;
    font-weight:700;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

</style>

""",
unsafe_allow_html=True
)


# -----------------------
# Header
# -----------------------

st.markdown(
"<div class='main-title'>📚 Oura</div>",
unsafe_allow_html=True
)

st.markdown(
"<div class='subtitle'>AI Document Assistant</div>",
unsafe_allow_html=True
)



# -----------------------
# Session memory
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# -----------------------
# Sidebar
# -----------------------

with st.sidebar:


    st.header("📄 Upload Documents")


    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type="pdf",
        accept_multiple_files=True
    )


    if st.button("Create Knowledge Base"):


        if uploaded_files:


            os.makedirs(
                "document loaders",
                exist_ok=True
            )


            pdf_paths = []


            for file in uploaded_files:


                path = os.path.join(
                    "document loaders",
                    file.name
                )


                with open(path,"wb") as f:

                    f.write(
                        file.getbuffer()
                    )


                pdf_paths.append(path)



            with st.spinner(
                "Creating vector database..."
            ):

                create_db(pdf_paths)


            st.success(
                "Knowledge Base Created!"
            )


        else:

            st.warning(
                "Please upload PDF files first."
            )



    st.divider()


    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.session_state.chat_history = []

        st.rerun()




# -----------------------
# Load Vector Database
# -----------------------

if os.path.exists("chroma-db"):


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


        (
        "system",

        """
You are Oura, an AI document assistant.

Answer only using the provided context.

Rules:

- Do not hallucinate.
- If information is missing say:
"I could not find the information to answer the question."

- Keep answers clear and concise.
- Use markdown formatting.
- Use conversation history for follow-up questions.

"""
        ),


        (
        "human",

        """
Conversation history:

{history}


Context:

{context}


Question:

{question}

"""
        )

    ])



else:


    st.info(
        "Upload documents and create a knowledge base first."
    )

    retriever = None



# -----------------------
# Display old messages
# -----------------------

for message in st.session_state.messages:


    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )




# -----------------------
# Chat
# -----------------------

query = st.chat_input(
    "Ask something about your documents..."
)



if query and retriever:


    # user message

    st.session_state.messages.append(
        {
            "role":"user",
            "content":query
        }
    )


    with st.chat_message("user"):

        st.markdown(query)



    with st.chat_message("assistant"):


        with st.spinner(
            "Thinking..."
        ):


            # retrieve documents

            docs = retriever.invoke(query)



            context = "\n\n".join(

                [
                    doc.page_content
                    for doc in docs
                ]

            )



            history = "\n".join(

                [
                    f"User: {msg['user']}\nAI:{msg['ai']}"

                    for msg in st.session_state.chat_history

                ]

            )



            final_prompt = prompt.invoke(

                {
                    "context":context,

                    "question":query,

                    "history":history

                }

            )



            response = llm.invoke(
                final_prompt
            )



            st.markdown(
                response.content
            )



            # -----------------------
            # Sources
            # -----------------------

            with st.expander(
                "📄 Sources Used"
            ):


                for i,doc in enumerate(
                    docs,
                    start=1
                ):


                    source = doc.metadata.get(
                        "source",
                        "unknown"
                    )


                    page = doc.metadata.get(
                        "page"
                    )


                    if page:

                        page += 1


                    st.write(
                        f"{i}. {source} - Page {page}"
                    )



            # -----------------------
            # Retrieved Context
            # -----------------------

            with st.expander(
                "🔎 Retrieved Context"
            ):


                for i,doc in enumerate(
                    docs,
                    start=1
                ):


                    st.write(
                        f"Chunk {i}"
                    )


                    st.write(
                        doc.page_content[:300]
                        + "..."
                    )


                    st.divider()



    # save memory

    st.session_state.chat_history.append(

        {
            "user":query,

            "ai":response.content
        }

    )


    st.session_state.messages.append(

        {
            "role":"assistant",

            "content":response.content
        }

    )