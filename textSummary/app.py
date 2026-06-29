import os
import re
from urllib.parse import urlparse, parse_qs

import validators
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="LangChain: Summarize Text From YouTube or Website",
    page_icon="📝",
    layout="centered"
)

st.title("📝 LangChain: Summarize Text From YouTube or Website")
st.subheader("Summarize any YouTube video or website URL")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🔑 API Key")
    user_groq_api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password"
    )

groq_api_key = user_groq_api_key or os.getenv("GROQ_API_KEY")

url = st.text_input(
    "Enter YouTube or Website URL",
    placeholder="Paste YouTube or website URL here..."
)

# -----------------------------
# Helpers
# -----------------------------
def extract_youtube_video_id(youtube_url: str) -> str | None:
    parsed_url = urlparse(youtube_url)

    hostname = parsed_url.hostname or ""

    # Shorts URL
    if "youtube.com" in hostname and "/shorts/" in parsed_url.path:
        return parsed_url.path.split("/shorts/")[-1].split("/")[0]

    # Normal YouTube URL
    if hostname in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    # Short youtu.be URL
    if hostname == "youtu.be":
        return parsed_url.path.lstrip("/").split("?")[0]

    # Fallback regex
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", youtube_url)
    if match:
        return match.group(1)

    return None


def is_youtube_url(input_url: str) -> bool:
    return "youtube.com" in input_url or "youtu.be" in input_url


def load_youtube_transcript(youtube_url: str):
    video_id = extract_youtube_video_id(youtube_url)

    if not video_id:
        raise ValueError("Could not extract YouTube video ID from the URL.")

    ytt_api = YouTubeTranscriptApi()

    fetched_transcript = ytt_api.fetch(
        video_id,
        languages=["en", "hi"]
    )

    transcript_text = " ".join(
        snippet.text for snippet in fetched_transcript
    )

    return [
        Document(
            page_content=transcript_text,
            metadata={
                "source": youtube_url,
                "video_id": video_id
            }
        )
    ]


# -----------------------------
# Prompt
# -----------------------------
prompt_template = """
Write a clear and concise summary of the following content.

Give the summary in simple language.

Content:
{text}

Summary:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

# -----------------------------
# App Logic
# -----------------------------
if st.button("Summarize Content"):
    if not groq_api_key:
        st.error("Please provide your Groq API key.")

    elif not url.strip():
        st.error("Please enter a URL.")

    elif not validators.url(url):
        st.error("Please enter a valid YouTube or website URL.")

    else:
        try:
            with st.spinner("Loading content..."):

                if is_youtube_url(url):
                    docs = load_youtube_transcript(url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        }
                    )
                    docs = loader.load()

                if not docs:
                    st.error("No content found from this URL.")
                    st.stop()

            with st.spinner("Splitting content..."):

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200
                )

                split_docs = splitter.split_documents(docs)

            with st.spinner("Summarizing content..."):

                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    groq_api_key=groq_api_key,
                    temperature=0,
                    max_tokens=500
                )

                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="map_reduce",
                    verbose=True
                )

                result = chain.invoke({
                    "input_documents": split_docs
                })

                st.success("Summary generated successfully!")
                st.write(result["output_text"])

        except Exception as e:
            st.error(f"Something went wrong: {e}")