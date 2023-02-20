import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.llm_math.base import LLMMathChain
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain import SQLDatabase, SQLDatabaseChain
from gpt_index import GPTSimpleVectorIndex, WikipediaReader
from sqlalchemy import create_engine

st.set_page_config(
    page_title="Demo", page_icon=":bird:"
)
st.header("Snowflake Demo")
st.write(
    "👋 This is a demo of connecting an LLM to a Snowflake database"
)

llm = OpenAI(temperature=0)

@st.cache_resource
def build_snowflake_chain():
    sf_url = "snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}".format(
        **st.secrets["snowflake"]
    )

    engine = create_engine(sf_url)
    sql_database = SQLDatabase(engine)

    st.write("❄️ Snowflake database connected")

    db_chain = SQLDatabaseChain(llm=llm, database=sql_database)
    return db_chain

db_chain = build_snowflake_chain()

tools = [
    Tool(
        name="Snowflake Transactions",
        func=lambda q: db_chain.run(q),
        description="Use this when you want to answer questions about customer orders, spending, purchases, and transactions. The input to this tool should be a complete english sentence.",
    ),
    Tool(
        "Calculator",
        LLMMathChain(llm=llm).run,
        "Useful for when you need to make any math calculations. Use this tool for any and all numerical calculations. The input to this tool should be a mathematical expression.",
    ),
]

if "generated" not in st.session_state:
    st.session_state["generated"] = ""

# memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True
)

user_input = st.text_input("", placeholder="Type your query here", key="input")

if user_input:
    if user_input == "":
        st.session_state["generated"] = ""
    else:
        output = agent_chain.run(input=user_input)
        st.session_state["generated"] = output

st.write(st.session_state["generated"])