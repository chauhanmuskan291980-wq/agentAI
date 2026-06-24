import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# LangSmith Configuration (Optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With Gemini"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's questions."),
        ("user", "Question: {question}")
    ]
)

# Function to Generate Response
from langchain_google_genai import ChatGoogleGenerativeAI

def generate_response(question, api_key, model_name, temperature):

    os.environ["GOOGLE_API_KEY"] = api_key

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain.invoke(
        {"question": question}
    )

# Streamlit UI
st.set_page_config(
    page_title="Gemini Q&A Chatbot",
    page_icon="🤖"
)

st.title("🤖 Q&A Chatbot using Gemini + LangChain")

# Sidebar
st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter your Gemini API Key",
    type="password"
)

model_name = st.sidebar.selectbox(
    "Select Gemini Model",
    [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash"
    ]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=50,
    max_value=1000,
    value=300
)

# Main Input
st.write("Ask me anything!")

user_input = st.text_input("You:")

if user_input:

    if not api_key:
        st.warning("Please enter your Gemini API Key.")
    else:
        try:
            response = generate_response(
                user_input,
                api_key,
                model_name,
                temperature
            )

            st.success(response)

        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    st.info("Enter a question to get started.")