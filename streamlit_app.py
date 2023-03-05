import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.llm_math.base import LLMMathChain
from langchain.llms import OpenAI
from langchain.utilities import PythonREPL
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain import SQLDatabase, SQLDatabaseChain
from gpt_index import GPTSimpleVectorIndex, WikipediaReader
from sqlalchemy import create_engine

# This is a comment

st.set_page_config(
    page_title="Demo", page_icon="🌩️"
)
st.subheader("Snowflake Demo")
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

    db_chain = SQLDatabaseChain(llm=llm, database=sql_database, return_direct=True, top_k=10)
    return db_chain

db_chain = build_snowflake_chain()

st.write("Examples you can try:")
st.write("""
    - Give me top 10 nations by order count, also return their order counts
    - Give me top 10 customers by order count, also return their order counts
    - Give me top 10 customer names by total order value, also get their total order values
""")

PROMPT_TEMPLATE = """
    When you are asked to retrieve data, always add a hard limit of 10 to all the queries.
    ###
    {query}
"""
prompt = PromptTemplate(
        input_variables=["query"],
        template=PROMPT_TEMPLATE,
    )

tools = [
    Tool(
        name="Snowflake Transactions",
        func=lambda q: db_chain.run(q),
        description="""Use this when you want to answer questions about customer orders, spending, purchases, and transactions. 
            The input to this tool should be a complete english sentence.
            Fetch only max 10 records always.
            I always need output as python data frame.""",
    ),
    Tool(
        "Calculator",
        LLMMathChain(llm=llm).run,
        """Useful for when you need to make any math calculations. 
        Use this tool for any and all numerical calculations. 
        The input to this tool should be a mathematical expression.""",
    ),
    Tool(
        "PythonREPL",
        PythonREPL().run,
        """Useful for running python code. Inout to this would be python code.
        This interface will only return things that are printed, 
        therefore if you want to use it to calculate an answer, 
        make sure to have it print out the answer.
        """,
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
        output = agent_chain.run(prompt.format(query=user_input))
        # output = agent_chain.run(input=user_input)
        st.session_state["generated"] = output

st.write(st.session_state["generated"])