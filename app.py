# import streamlit as st
# from streamlit_chat import message
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# from agents.router_agent import router_agent
# from agents.recommender_agent import recommend_query
# import asyncio
# import time
# from langchain_chroma import Chroma
# from langchain_community.llms import HuggingFaceHub
# from langchain_ollama.llms import OllamaLLM
# from dotenv import load_dotenv
# from langchain_ollama import ChatOllama
# import os

# load_dotenv()


# st.title("🤖 Healthcare E-Commerce Chatbot ")
# st.markdown("""This AI-powered chatbot is designed to enhance the 🏥 healthcare e-commerce experience by providing users with accurate **product information** 📋, seamless 💰 **price comparison**, effortless 🛒 **ordering**, and 🎯 personalized **recommendations.** It leverages specialized AI agents to ensure a smooth, efficient, and intelligent shopping journey.""")
# st.subheader("",divider="rainbow")


# def main():
#     # st.title("🤖 FracsNet Chatbot")
#     # llm = ChatOllama(
#     #     model="nemotron-mini",
#     #     temperature=0,
#     #     num_predict=256,
#     # )

#     llm = ChatGroq(model_name="llama-3.1-8b-instant")

#     @st.cache_resource
#     def load_vectordb():
#         embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#         try:
#             loaded_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
#             return loaded_db
#         except Exception as e:
#             st.error(f"Error loading vector database: {e}")
#             return None
    

#     # @st.cache_data
#     def response_generator(vectordb, query):
#         # template = """Use the following pieces of context to answer the question at the end. 
#         # If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. {context} Question: {question} Helpful Answer:"""

#         template = """You are an AI assistant specializing in healthcare e-commerce. Use the provided context to answer the user's question accurately and concisely. 

#         **Guidelines:**
#         - **Be precise:** Limit your response to a maximum of three sentences.
#         - **Stay factual:** If the answer is unknown, state "I don't know" rather than guessing.
#         - **Be structured:** Use clear and direct language.

#         **Context:**  
#         {context}

#         **User's Question:**  
#         {question}  

#         **Your Expert Response:**"""

#         QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

#         qa_chain = RetrievalQA.from_chain_type(llm, 
#                                             retriever=vectordb.as_retriever(), 
#                                             return_source_documents=True, 
#                                             chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

#         ans = qa_chain.invoke(query)
#         return ans["result"]

#     vectordb = load_vectordb()
#     if vectordb:
#         # Initialize session state variables
#         if "messages" not in st.session_state:
#             st.session_state.messages = []
#         if "current_recommendations" not in st.session_state:
#             st.session_state.current_recommendations = []
#         if "clicked_recommendation" not in st.session_state:
#             st.session_state.clicked_recommendation = None

#         # Display chat history
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#         # Process clicked recommendation if exists
#         if st.session_state.clicked_recommendation:
#             rec_query = st.session_state.clicked_recommendation
#             st.session_state.clicked_recommendation = None  # Reset after processing
            
#             # Add recommendation as user message
#             st.session_state.messages.append({"role": "user", "content": rec_query})
#             with st.chat_message("user"):
#                 st.markdown(rec_query)
            
#             # Generate and display response for recommendation
#             with st.spinner("Generating response for recommendation..."):
#                 rec_response = response_generator(vectordb, rec_query)
#                 with st.chat_message("assistant"):
#                     st.markdown(rec_response)
#                 st.session_state.messages.append({"role": "assistant", "content": rec_response})
            
#             # Generate new recommendations for the recommendation query
#             st.session_state.current_recommendations = recommend_query(rec_query)
#             st.rerun()

#         # Handle user input
#         if query := st.chat_input("What is up?"):
#             # Add user message to chat
#             st.session_state.messages.append({"role": "user", "content": query})
#             with st.chat_message("user"):
#                 st.markdown(query)

#             start_time = time.time()
#             routed_agent = router_agent(query)
#             st.write(routed_agent)

#             if routed_agent["intent"] == "INFO":
#                 # Generate main response
#                 with st.spinner("Generating response..."):
#                     response = response_generator(vectordb, query)
#                     end_time = time.time()
#                     time_taken = end_time - start_time
#                     response_with_time = f"{response}\n\n*(Response generated in {time_taken:.2f} seconds)*"

#                 # Add assistant response to chat
#                 with st.chat_message("assistant"):
#                     st.markdown(response_with_time)
#                 st.session_state.messages.append({"role": "assistant", "content": response_with_time})

#                 # Generate and store recommendations
#                 st.session_state.current_recommendations = recommend_query(query)

#             else:
#                 response = f"Query will be routed to the {routed_agent['intent']} agent..."
#                 with st.chat_message("assistant"):
#                     st.markdown(response)
#                 st.session_state.messages.append({"role": "assistant", "content": response})

#         # Display recommendations in expander
#         if st.session_state.current_recommendations:
#             with st.expander("📌 **Check out these relevant recommendations...**"):
#                 for rec in st.session_state.current_recommendations:
#                     if st.button(rec, key=f"rec_{rec}"):
#                         st.session_state.clicked_recommendation = rec
#                         st.rerun()

# if __name__ == "__main__":
#     main()









#######################################
#    Question Recommendation Added    #
#######################################

# import streamlit as st
# from agents.recommender_agent import recommend_query
# # Initialize session state for selected recommendation
# if "selected_recommendation" not in st.session_state:
#     st.session_state.selected_recommendation = None

# recommendations = [
#     "What are the prices of Dolo?",
#     "I wanna order Dolo.",
#     "What are the side effects of Dolo?"
# ]

# # Expander for recommendations
# with st.expander("📌 **Check out these relevant recommendations...**"):
#     for rec in recommendations:
#         if st.button(rec):
#             st.session_state.selected_recommendation = rec  # Store selected recommendation
#             st.rerun()  # Rerun to reflect changes outside expander

# # Display selected recommendation outside the expander
# if st.session_state.selected_recommendation:
#     st.markdown(f"📝 You selected: **{st.session_state. selected_recommendation}**")





######################################
#    Product Recommendation Added    #
######################################

# with st.expander("📌 **You may also like ...**"):
    
#     # Create three columns
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.image("assets/youm.png", caption="youm vitamin")
#     with col2:
#         st.image("assets/herbal_tea.png", caption="Herbal Tea")
#     with col3:
#         st.image("assets/protein_powder.png", caption="Protein Powder")




######################################
#    without adding login page       #
######################################



# import streamlit as st
# from streamlit_chat import message
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# from agents.router_agent import router_agent
# from agents.recommender_agent import recommend_query
# import asyncio
# import time
# from langchain_chroma import Chroma
# from langchain_community.llms import HuggingFaceHub
# from langchain_ollama.llms import OllamaLLM
# from dotenv import load_dotenv
# from langchain_ollama import ChatOllama
# import os
# from agents.order_agent import OrderTakingAssistant
# load_dotenv()


# st.title("🤖 Healthcare E-Commerce Chatbot ")
# st.markdown("""This AI-powered chatbot is designed to enhance the 🏥 healthcare e-commerce experience by providing users with accurate **product information** 📋, seamless 💰 **price comparison**, effortless 🛒 **ordering**, and 🎯 personalized **recommendations.** It leverages specialized AI agents to ensure a smooth, efficient, and intelligent shopping journey.""")
# st.subheader("",divider="rainbow")


# def main():
#     # Initialize LLM
#     llm = ChatGroq(model_name="llama-3.1-8b-instant")

#     @st.cache_resource
#     def load_vectordb():
#         embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#         try:
#             loaded_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
#             return loaded_db
#         except Exception as e:
#             st.error(f"Error loading vector database: {e}")
#             return None
    
#     def response_generator(vectordb, query):
#         template = """You are an AI assistant specializing in healthcare e-commerce. Use the provided context to answer the user's question accurately and concisely. 

#         **Guidelines:**
#         - **Be precise:** Limit your response to a maximum of three sentences.
#         - **Stay factual:** If the answer is unknown, state "I don't know" rather than guessing.
#         - **Be structured:** Use clear and direct language.

#         **Context:**  
#         {context}

#         **User's Question:**  
#         {question}  

#         **Your Expert Response:**"""

#         QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

#         qa_chain = RetrievalQA.from_chain_type(llm, 
#                                             retriever=vectordb.as_retriever(), 
#                                             return_source_documents=True, 
#                                             chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

#         ans = qa_chain.invoke(query)
#         return ans["result"]

#     vectordb = load_vectordb()
#     if vectordb:
#         # Initialize session state variables
#         if "messages" not in st.session_state:
#             st.session_state.messages = []
#         # if "current_recommendations" not in st.session_state:
#         #     st.session_state.current_recommendations = []
#         # if "clicked_recommendation" not in st.session_state:
#         #     st.session_state.clicked_recommendation = None
#         if "order_mode" not in st.session_state:
#             st.session_state.order_mode = False
#         if "order_assistant" not in st.session_state:
#             st.session_state.order_assistant = None

#         # Display chat history
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#         # Process clicked recommendation if exists
#         # if st.session_state.clicked_recommendation:
#         #     rec_query = st.session_state.clicked_recommendation
#         #     st.session_state.clicked_recommendation = None  # Reset after processing
            
#         #     # Add recommendation as user message
#         #     st.session_state.messages.append({"role": "user", "content": rec_query})
#         #     with st.chat_message("user"):
#         #         st.markdown(rec_query)
            
#         #     # Generate and display response for recommendation
#         #     with st.spinner("Generating response for recommendation..."):
#         #         rec_response = response_generator(vectordb, rec_query)
#         #         with st.chat_message("assistant"):
#         #             st.markdown(rec_response)
#         #         st.session_state.messages.append({"role": "assistant", "content": rec_response})
            
#         #     # Generate new recommendations for the recommendation query
#         #     st.session_state.current_recommendations = recommend_query(rec_query)
#         #     st.rerun()

#         # Handle user input
#         if query := st.chat_input("What is up?"):
#             # Add user message to chat
#             st.session_state.messages.append({"role": "user", "content": query})
#             with st.chat_message("user"):
#                 st.markdown(query)

#             # If in order mode, process with order agent
#             if st.session_state.order_mode and st.session_state.order_assistant:
#                 # Process user input with the order assistant
#                 response = st.session_state.order_assistant.process_input(query)
                
#                 # Display assistant response
#                 with st.chat_message("assistant"):
#                     st.markdown(response)
                
#                 # Add assistant response to chat history
#                 st.session_state.messages.append({"role": "assistant", "content": response})
                
#                 # Check if order is complete
#                 if st.session_state.order_assistant.is_order_complete():
#                     # Display order details
#                     order_details = st.session_state.order_assistant.get_order_details()
#                     with st.chat_message("assistant"):
#                         st.markdown("✅ Order Complete! Final Details:")
#                         st.json(order_details)
                    
#                     # Exit order mode
#                     st.session_state.order_mode = False
#                     st.session_state.order_assistant = None
                    
#                     # Add completion message to chat history
#                     completion_msg = "Your order has been successfully placed. Is there anything else I can help you with?"
#                     st.session_state.messages.append({"role": "assistant", "content": completion_msg})
#                     with st.chat_message("assistant"):
#                         st.markdown(completion_msg)
#             else:
#                 # Use intent router for normal mode
#                 start_time = time.time()
#                 routed_agent = router_agent(query)
                
#                 if routed_agent["intent"] == "INFO":
#                     # Generate main response
#                     with st.spinner("Generating response..."):
#                         response = response_generator(vectordb, query)
#                         end_time = time.time()
#                         time_taken = end_time - start_time
#                         response_with_time = f"{response}\n\n*(Response generated in {time_taken:.2f} seconds)*"

#                     # Add assistant response to chat
#                     with st.chat_message("assistant"):
#                         st.markdown(response_with_time)
#                     st.session_state.messages.append({"role": "assistant", "content": response_with_time})

#                     # Generate and store recommendations
#                     # st.session_state.current_recommendations = recommend_query(query)

#                 elif routed_agent["intent"] == "ORDER":
#                     # Initialize order assistant and enter order mode
#                     st.session_state.order_assistant = OrderTakingAssistant()
#                     st.session_state.order_mode = True
                    
#                     # Add welcome message to chat
#                     welcome_msg = "Welcome to the ordering system 🛒! I'll help you place your order step by step."
#                     with st.chat_message("assistant"):
#                         st.markdown(welcome_msg)
#                     st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                    
#                     # Get first prompt from the order assistant
#                     first_prompt = st.session_state.order_assistant.get_next_prompt()
#                     with st.chat_message("assistant"):
#                         st.markdown(first_prompt)
#                     st.session_state.messages.append({"role": "assistant", "content": first_prompt})
#                 else:
#                     # Handle other intents
#                     response = f"Query will be routed to the {routed_agent['intent']} agent..."
#                     with st.chat_message("assistant"):
#                         st.markdown(response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})

#         # Display recommendations in expander (only when not in order mode)
#         # if not st.session_state.order_mode and st.session_state.current_recommendations:
#         #     with st.expander("📌 **Check out these relevant recommendations...**"):
#         #         for rec in st.session_state.current_recommendations:
#         #             if st.button(rec, key=f"rec_{rec}"):
#         #                 st.session_state.clicked_recommendation = rec
#         #                 st.rerun()

# if __name__ == "__main__":
#     main()




######################################################
#    adding login page but removed recommendation    #
######################################################

import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from agents.router_agent import router_agent
import time
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os
from agents.order_agent import OrderTakingAssistant
load_dotenv()

# Initialize session state variables for login and chat
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_type" not in st.session_state:
    st.session_state.user_type = None
if "username" not in st.session_state:
    st.session_state.username = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "order_mode" not in st.session_state:
    st.session_state.order_mode = False
if "order_assistant" not in st.session_state:
    st.session_state.order_assistant = None

# Mock user database - in a real application, you would use a secure database
users = {
    "doctor1": "password1",
    "doctor2": "password2",
    "patient1": "password1",
    "patient2": "password2",
}

def login_page():
    st.title("🏥 Healthcare E-Commerce Platform")
    st.subheader("Login", divider="rainbow")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        user_type = st.selectbox("Select User Type", ["Doctor", "Patient"])
        submit = st.form_submit_button("Login")
        
        if submit:
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_type = user_type
                st.success(f"Successfully logged in as {user_type}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    st.markdown("---")
    st.markdown("**Don't have an account?**")
    if st.button("Register Now"):
        st.info("Registration functionality would be implemented here in a real application.")

def chatbot_interface():
    st.title("🤖 Healthcare E-Commerce Chatbot")
    
    # Display user information
    st.sidebar.success(f"Logged in as: {st.session_state.username} ({st.session_state.user_type})")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_type = None
        st.session_state.messages = []
        st.session_state.order_mode = False
        st.session_state.order_assistant = None
        st.rerun()
    
    # Welcome message based on user type
    if st.session_state.user_type == "Doctor":
        welcome_text = """This AI-powered chatbot is designed to enhance your healthcare e-commerce experience by providing you with accurate **product information** 📋, seamless 💰 **price comparison**, and effortless 🛒 **ordering** for your practice and patients. It has specialized features for medical professionals."""
    else:  # Patient
        welcome_text = """This AI-powered chatbot is designed to enhance your healthcare e-commerce experience by providing you with accurate **product information** 📋 and seamless 💰 **price comparison** based on your health needs. We prioritize your health and privacy. *Note: Product ordering is only available for healthcare providers.*"""
        
    st.markdown(welcome_text)
    st.subheader("", divider="rainbow")

    # Main chatbot functionality
    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    @st.cache_resource
    def load_vectordb():
        embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        try:
            loaded_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
            return loaded_db
        except Exception as e:
            st.error(f"Error loading vector database: {e}")
            return None
    
    def response_generator(vectordb, query):
        # Customize template based on user type
        if st.session_state.user_type == "Doctor":
            template = """You are an AI assistant specializing in healthcare e-commerce for medical professionals. Use the provided context to answer the doctor's question accurately and provide medically relevant information.

            **Guidelines:**
            - **Be precise:** Provide clinical details appropriate for a medical professional.
            - **Stay factual:** If the answer is unknown, state "I don't know" rather than guessing.
            - **Be structured:** Use professional medical terminology where appropriate.

            **Context:**  
            {context}

            **Doctor's Question:**  
            {question}  

            **Your Medical Expert Response:**"""
        else:  # Patient
            template = """You are an AI assistant specializing in healthcare e-commerce for patients. Use the provided context to answer the patient's question in patient-friendly language.

            **Guidelines:**
            - **Be accessible:** Use clear, non-technical language suitable for patients.
            - **Stay factual:** If the answer is unknown, state "I don't know" rather than guessing.
            - **Be helpful:** Provide general health guidance when appropriate.

            **Context:**  
            {context}

            **Patient's Question:**  
            {question}  

            **Your Patient-Friendly Response:**"""

        # template = """Use the following pieces of context to answer the question at the end. 
        # If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. {context} Question: {question} Helpful Answer:"""


        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        qa_chain = RetrievalQA.from_chain_type(llm, 
                                            retriever=vectordb.as_retriever(), 
                                            return_source_documents=True, 
                                            chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

        ans = qa_chain.invoke(query)
        return ans["result"]

    vectordb = load_vectordb()
    if vectordb:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input
        placeholder_text = "What medications are you looking for?" if st.session_state.user_type == "Patient" else "Ask about products, pricing, or clinical information..."
        
        if query := st.chat_input(placeholder_text):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # If in order mode, process with order agent
            if st.session_state.order_mode and st.session_state.order_assistant:
                # Process user input with the order assistant
                response = st.session_state.order_assistant.process_input(query)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Check if order is complete
                if st.session_state.order_assistant.is_order_complete():
                    # Display order details
                    order_details = st.session_state.order_assistant.get_order_details()
                    with st.chat_message("assistant"):
                        st.markdown("✅ Order Complete! Final Details:")
                        st.json(order_details)
                    
                    # Exit order mode
                    st.session_state.order_mode = False
                    st.session_state.order_assistant = None
                    
                    # Add completion message to chat history
                    completion_msg = "Your order has been successfully placed. Is there anything else I can help you with?"
                    st.session_state.messages.append({"role": "assistant", "content": completion_msg})
                    with st.chat_message("assistant"):
                        st.markdown(completion_msg)
            else:
                # Use intent router for normal mode
                start_time = time.time()
                routed_agent = router_agent(query)
                st.write(routed_agent)
                
                if routed_agent["intent"] == "INFO":
                    # Generate main response
                    with st.spinner("Generating response..."):
                        response = response_generator(vectordb, query)
                        end_time = time.time()
                        time_taken = end_time - start_time
                        response_with_time = f"{response}\n\n*(Response generated in {time_taken:.2f} seconds)*"

                    # Add assistant response to chat
                    with st.chat_message("assistant"):
                        st.markdown(response_with_time)
                    st.session_state.messages.append({"role": "assistant", "content": response_with_time})

                elif routed_agent["intent"] == "ORDER":
                    # Check user type for order permission
                    if st.session_state.user_type == "Doctor":
                        # Initialize order assistant and enter order mode
                        st.session_state.order_assistant = OrderTakingAssistant()
                        st.session_state.order_mode = True
                        
                        # Add welcome message to chat
                        welcome_msg = "Welcome to the ordering system, Doctor! I'll help you place your order step by step."
                        with st.chat_message("assistant"):
                            st.markdown(welcome_msg)
                        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                        
                        # Get first prompt from the order assistant
                        first_prompt = st.session_state.order_assistant.get_next_prompt()
                        with st.chat_message("assistant"):
                            st.markdown(first_prompt)
                        st.session_state.messages.append({"role": "assistant", "content": first_prompt})
                    else:
                        # Patient tried to order - notify that ordering is restricted
                        restricted_msg = "🚫 Oops! Ordering products is not applixcable for **patients**.If you need medications 💊 or healthcare products, please consult with your healthcare provider.👩‍⚕️✅"
                        with st.chat_message("assistant"):
                            st.markdown(restricted_msg)
                        st.session_state.messages.append({"role": "assistant", "content": restricted_msg})
                        
                else:
                    # Handle other intents
                    response = f"Query will be routed to the {routed_agent['intent']} agent..."
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                        
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        chatbot_interface()

if __name__ == "__main__":
    main()

