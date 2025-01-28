# import streamlit as st
# from router import router_agent
# from langchain_ollama import ChatOllama
# import time 

# @st.cache_resource
# def llm_ans(query):
#     llm = ChatOllama(
#         model = "nemotron-mini",
#         temperature = 0.2,
#         num_predict = 256,
#         # other params ...
#     )
#     res = llm.invoke(query).content

#     return res


# def gen_info_agent():
#     pass

# def order_agent():
#     pass

# def comparision_agent():
#     pass

# def recommandation_agent():
#     pass


# queries = [
#     "I want to buy vitamin D supplements.",
#     "Can you tell me the benefits of fish oil?",
#     "Add 2 bottles of multivitamins to my cart.",
#     "Which is better, vitamin C or zinc for immunity?",
#     "Recommend something for joint pain relief.",
#     "How does omega-3 help the heart?",
# ]


# # query = st.text_input("Enter your query...")

# # if query:
# #     route = route_info(query)
# #     st.write(route)
# #     r = route["primary_intent"]

# #     st.success(f"This Query will be routed to the {r} agent")

# #     if r== "INFO":

# #         st.markdown(llm_ans(query))


# for q in queries:
#     st.write(q)
#     st.write(router_agent(q))
#     st.divider()

    



import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from router_agent import router_agent
import asyncio
import time  # For time tracking
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

st.title("🤖 FracsNet Chatbot")
llm = ChatGroq(model_name="Llama3-8b-8192")
# from langchain_groq import ChatGroq
# Initialize Hugging Face model
# llm = HuggingFaceHub(
#     repo_id="google/flan-t5-base",  # You can change this to any other model
#     model_kwargs={"temperature": 0.7, "max_length": 512}
# )


# llm = OllamaLLM(model="mistral")



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

async def response_generator(vectordb, query):
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. {context} Question: {question} Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    
    result = await asyncio.to_thread(qa_chain, {"query": query})
    return result["result"]

vectordb = load_vectordb()
if vectordb:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if query := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": query})

        

        with st.chat_message("user"):
            st.markdown(query) 
        
        # Start time tracking
        start_time = time.time()
        if router_agent(query)["intent"]=="INFO":
            with st.spinner("Generating response..."):
                response = asyncio.run(response_generator(vectordb, query))
        
        else:
            response = router_agent(query)["intent"]
            
        # End time tracking
        end_time = time.time()
        time_taken = end_time - start_time
        
        # Format the response with time taken
        response_with_time = f"{response}\n\n*(Response generated in {time_taken:.2f} seconds)*"
        
        with st.chat_message("assistant"):
            st.markdown(response_with_time)
        st.session_state.messages.append({"role": "assistant", "content": response_with_time})
