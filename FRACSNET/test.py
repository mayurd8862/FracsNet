# import streamlit as st
# from streamlit_chat import message
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# from agents.router_agent import router_agent
# import asyncio
# import time  # For time tracking
# from langchain_chroma import Chroma
# from langchain_community.llms import HuggingFaceHub
# from langchain_ollama.llms import OllamaLLM
# from dotenv import load_dotenv
# from langchain_ollama import ChatOllama
# load_dotenv()

# st.title("🤖 FracsNet Chatbot")
# llm = ChatGroq(model_name="Llama3-8b-8192")

# llm = ChatOllama(
#     model = "nemotron-mini",
#     temperature = 0,
#     num_predict = 256,
#     # other params ...
# )

# @st.cache_resource
# def load_vectordb():
#     # Initialize the embedding model
#     embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
#     try:
#         # Load the previously saved Chroma vector store
#         loaded_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
#         return loaded_db
#     except Exception as e:
#         print(f"Error loading vector database: {e}")
#         return None

# async def response_generator(vectordb, query):
#     template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. {context} Question: {question} Helpful Answer:"""
#     QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
#     qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    
#     result = await asyncio.to_thread(qa_chain, {"query": query})
#     return result["result"]

# vectordb = load_vectordb()
# if vectordb:
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
    
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     if query := st.chat_input("What is up?"):
#         st.session_state.messages.append({"role": "user", "content": query})

#         with st.chat_message("user"):
#             st.markdown(query) 
        
#         start_time = time.time()
#         routed_agent = router_agent(query)
#         st.write(routed_agent)
        
#         # Start time tracking
        
#         if routed_agent["intent"]=="INFO":
#             with st.spinner("Generating response..."):
#                 response = asyncio.run(response_generator(vectordb, query))
        
#         else:
#             response = "Query will be routed to the " + routed_agent["intent"] + " agent..."

            
#         # End time tracking
#         end_time = time.time()
#         time_taken = end_time - start_time
        
#         # Format the response with time taken
#         response_with_time = f"{response}\n\n*(Response generated in {time_taken:.2f} seconds)*"
        
#         with st.chat_message("assistant"):
#             st.markdown(response_with_time)
#         st.session_state.messages.append({"role": "assistant", "content": response_with_time})

#         # Add a recommendation box after each query-response pair
#         with st.expander("Here are some recommendations based on your query ..."):
#             # st.write("Here are some recommendations based on your query:")
#             # You can add your recommendation logic here
#             # For example, you can use the response to generate recommendations
#             recommendations = [
#                 "Recommendation 1: Explore related topics.",
#                 "Recommendation 2: Check out our documentation.",
#                 "Recommendation 3: Contact support for more help."
#             ]
#             for rec in recommendations:
#                 st.write(rec)


# import streamlit as st
# recommendations = [
#     "what are the prices of Dolo?",
#     "I wanna order dolo.",
#     "What are the side effects of dolo?"
# ]
# with st.expander("📌 **Check out these relevant recommendations...**"):
#     for rec in recommendations:
#         if st.button(rec):
#             st.markdown(rec)



#######################################
#    Question Recommendation Added    #
#######################################

import streamlit as st

# Initialize session state for selected recommendation
if "selected_recommendation" not in st.session_state:
    st.session_state.selected_recommendation = None

recommendations = [
    "What are the prices of Dolo?",
    "I wanna order Dolo.",
    "What are the side effects of Dolo?"
]

# Expander for recommendations
with st.expander("📌 **Check out these relevant recommendations...**"):
    for rec in recommendations:
        if st.button(rec):
            st.session_state.selected_recommendation = rec  # Store selected recommendation
            st.rerun()  # Rerun to reflect changes outside expander

# Display selected recommendation outside the expander
if st.session_state.selected_recommendation:
    st.markdown(f"📝 You selected: **{st.session_state. selected_recommendation}**")




######################################
#    Product Recommendation Added    #
######################################

with st.expander("📌 **You may also like ...**"):
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("assets/youm.png", caption="youm vitamin")
    with col2:
        st.image("assets/herbal_tea.png", caption="Herbal Tea")
    with col3:
        st.image("assets/protein_powder.png", caption="Protein Powder")
