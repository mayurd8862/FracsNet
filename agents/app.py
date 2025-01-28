import streamlit as st
from router import route_info
from langchain_ollama import ChatOllama
import time 

@st.cache_resource
def llm_ans(query):
    llm = ChatOllama(
        model = "nemotron-mini",
        temperature = 0.2,
        num_predict = 256,
        # other params ...
    )
    res = llm.invoke(query).content

    return res


def gen_info_agent():
    pass

def order_agent():
    pass

def comparision_agent():
    pass

def recommandation_agent():
    pass


query = st.text_input("Enter your query...")

if query:
    route = route_info(query)
    st.write(route)
    r = route["primary_intent"]

    st.success(f"This Query will be routed to the {r} agent")

    if r== "INFO":

        st.markdown(llm_ans(query))


    