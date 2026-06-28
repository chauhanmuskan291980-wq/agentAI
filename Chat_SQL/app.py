import streamlit as st
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from urllib.parse import quote_plus
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🧠")
st.title("LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = [
    "Use SQLite 3 Database - student.db",
    "Connect to your MySQL Database"
]

selected_opt = st.sidebar.radio(
    label="Choose the DB you want to chat with",
    options=radio_opt
)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input("GROQ API Key", type="password").strip()

if not api_key:
    st.warning("Please add the Groq API key")
    st.stop()


llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    streaming=True
)


@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()

        def creator(): return sqlite3.connect(
            f"file:{dbfilepath}?mode=ro",
            uri=True
        )

        return SQLDatabase(
            create_engine(
                "sqlite:///",
                creator=creator
            )
        )

    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_db):
            st.error("Please provide all MySQL connection details")
            st.stop()
        return SQLDatabase(
            create_engine(
                f"mysql+mysqlconnector://{mysql_user}@{mysql_host}/{mysql_db}"
            )
        )


if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)


toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_executor_kwargs={
        "return_intermediate_steps": True
    }
)


user_query = st.chat_input("Ask anything from your database")

if user_query:
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())

        try:
            response = agent.invoke(
                {"input": user_query},
                callbacks=[streamlit_callback]
            )

            st.write(response["output"])

        except Exception as e:
            st.error(f"Error: {e}")
             
        # Show SQL trace / citation
        with st.expander("View SQL query and database steps"):
            for step in response.get("intermediate_steps", []):
                action = step[0]
                observation = step[1]

                st.markdown(f"**Tool used:** `{action.tool}`")

                if action.tool_input:
                    st.markdown("**Input:**")
                    st.code(str(action.tool_input), language="sql")

                st.markdown("**Result:**")
                st.write(observation)