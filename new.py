import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import asyncio
import time  # For time tracking
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

st.title("🤖 FracsNet Chatbot")
# llm = ChatGroq(model_name="Llama3-8b-8192")

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "nemotron-mini",
    temperature = 0.8,
    num_predict = 256,
    # other params ...
)

@st.cache_resource
def load_vectordb():
    # Initialize the embedding model
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        # Load the previously saved Chroma vector store
        loaded_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
        return loaded_db
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None
    


# Add recommendations to session state
if "recommendations_history" not in st.session_state:
    st.session_state.recommendations_history = []


def generate_recommendations(query):
    # Logic for generating recommendations based on the query/response
    # This is a placeholder; replace with actual logic
    recommendations = [
        "Recommendation 1: Explore related medications.",
        "Recommendation 2: Read research materials for deeper insights.",
        "Recommendation 3: Consider lifestyle changes to complement medication."
    ]
    return recommendations


async def response_generator(vectordb, query):
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. {context} Question: {question} Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    
    result = await asyncio.to_thread(qa_chain, {"query": query})
    return result["result"]
import streamlit as st
# ... (previous imports remain the same)

import streamlit as st
# ... (previous imports)

if "messages" not in st.session_state:
    st.session_state.messages = []


vectordb = load_vectordb()
if vectordb:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("recommendations"):
                with st.container():
                    st.markdown("**Recommendations:**")
                    for rec in message["recommendations"]:
                        st.markdown(f"- {rec}")
    
    if query := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        start_time = time.time()
        with st.spinner("Generating response..."):
            response = asyncio.run(response_generator(vectordb, query))
        end_time = time.time()
        
        response_with_time = f"{response}\n\n*(Response generated in {end_time - start_time:.2f} seconds)*"
        recommendations = generate_recommendations(query)
        
        with st.chat_message("assistant"):
            st.markdown(response_with_time)
            with st.container():
                st.markdown("**Recommendations:**")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
                st.markdown("**--------------------------------------**")
        
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_with_time,
            "recommendations": recommendations
        })