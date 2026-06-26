import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# --------------------------------------------------
# Load Environment
# --------------------------------------------------

load_dotenv()

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------

st.set_page_config(
    page_title="Conversational PDF RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Conversational RAG with PDF")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
    )

    session_id = st.text_input(
        "Session ID",
        value="default",
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf",
    )

if not groq_api_key:
    st.warning("Enter your Groq API Key.")
    st.stop()

# --------------------------------------------------
# Embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
)

# --------------------------------------------------
# Session Store
# --------------------------------------------------

if "store" not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Chat History Function
# --------------------------------------------------

def get_session_history(session: str) -> BaseChatMessageHistory:

    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()

    return st.session_state.store[session]


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

if uploaded_file:

    if (
        "uploaded_name" not in st.session_state
        or st.session_state.uploaded_name != uploaded_file.name
    ):

        st.session_state.uploaded_name = uploaded_file.name

        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

            tmp.write(uploaded_file.read())

            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        splits = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db",
        )

        st.session_state.vectorstore = vectorstore

        os.remove(pdf_path)

# --------------------------------------------------
# Stop if no PDF
# --------------------------------------------------

if "vectorstore" not in st.session_state:

    st.info("Upload a PDF to begin.")

    st.stop()

# --------------------------------------------------
# Retriever
# --------------------------------------------------

retriever = st.session_state.vectorstore.as_retriever(
    search_kwargs={"k":4}
)

# --------------------------------------------------
# Context Prompt
# --------------------------------------------------

contextualize_system = """
Given the chat history and latest user question,
rewrite the question so it can be understood without
the chat history.

Do NOT answer.

Only rewrite if necessary.
"""

context_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_retriever = create_history_aware_retriever(
    llm,
    retriever,
    context_prompt,
)

# --------------------------------------------------
# QA Prompt
# --------------------------------------------------

qa_system = """
You are an AI assistant.

Use the retrieved context to answer.

If the answer isn't in the context,
say you don't know.

Answer in three concise sentences.

Context:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)

rag_chain = create_retrieval_chain(
    history_retriever,
    question_chain,
)

conversation = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --------------------------------------------------
# Display Chat
# --------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# User Question
# --------------------------------------------------

question = st.chat_input("Ask anything about the PDF...")

if question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):

        response = conversation.invoke(
            {
                "input":question,
            },
            config={
                "configurable":{
                    "session_id":session_id
                }
            },
        )

    answer = response["answer"]

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

    with st.expander("Retrieved Context"):

        for i, doc in enumerate(response["context"], 1):

            st.markdown(f"### Chunk {i}")

            st.write(doc.page_content)

            st.divider()