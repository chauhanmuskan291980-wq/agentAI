import os
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# -------------------- Page Config --------------------

st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper Q&A with Groq")
st.markdown("---")


# -------------------- Load Environment --------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()


# -------------------- LLM --------------------

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192",
)


# -------------------- Prompt --------------------

prompt = ChatPromptTemplate.from_template(
    """
Answer the user's question using ONLY the provided context.

If the answer is not available in the context, say:

"I couldn't find the answer in the provided documents."

<context>
{context}
</context>

Question:
{input}
"""
)


# -------------------- Vector DB --------------------

def create_vector_embedding():

    if "vectors" in st.session_state:
        st.success("Vector database already exists.")
        return

    with st.spinner("Loading and embedding documents..."):

        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        loader = PyPDFDirectoryLoader("research_paper")

        docs = loader.load()
        st.write(f"Documents Loaded: {len(docs)}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        final_documents = splitter.split_documents(docs)

        vectors = FAISS.from_documents(
            final_documents,
            embeddings
        )

        st.session_state.embeddings = embeddings
        st.session_state.vectors = vectors
        st.session_state.documents = final_documents

    st.success("Vector Database Created Successfully!")


# -------------------- Sidebar --------------------

with st.sidebar:

    st.header("Settings")

    if st.button("Create Vector Database"):
        create_vector_embedding()


# -------------------- User Input --------------------

user_prompt = st.text_input(
    "Ask a question about your research papers:"
)


# -------------------- Retrieval --------------------

if user_prompt:

    if "vectors" not in st.session_state:
        st.warning("Please create the Vector Database first.")
        st.stop()

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.time()

    response = retrieval_chain.invoke(
        {"input": user_prompt}
    )

    elapsed = time.time() - start

    st.markdown("## Answer")

    st.write(response["answer"])

    st.caption(f"Response Time: {elapsed:.2f} seconds")

    with st.expander("Retrieved Chunks"):

        for i, doc in enumerate(response["context"], start=1):

            st.markdown(f"### Chunk {i}")

            st.write(doc.page_content)

            st.divider()