from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With Ollama"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, engine, temperature):

    print("Question:", question)
    print("Model:", engine)

    llm = OllamaLLM(
        model=engine,
        temperature=temperature
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    answer = chain.invoke({"question": question})

    print("Answer:", answer)

    return answer



st.set_page_config(
    page_title="Ollama Chatbot",
    page_icon="🤖"
)

st.title("🤖 Q&A Chatbot using Ollama + LangChain")

st.sidebar.title("Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "llama3.2"
    ]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

user_input = st.text_input("Ask a question")

if user_input:
    try:
        response = generate_response(
            user_input,
            model_name,
            temperature
        )

        st.write(response)

    except Exception as e:
        st.error(str(e))