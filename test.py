
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
from agents.summarization_agent import summary_agent  
from agents.validator_agent import validator_agent
from agents.send_email import send_order_email,OTP_verification_email
from mongodb import register_user, login_user,get_user_details,place_order,get_user_orders
from src.database import get_product_image, get_product_names
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import random
from langchain_ollama import ChatOllama
import os
import pandas as pd
load_dotenv()
import uuid
from datetime import datetime, timedelta
import streamlit as st
from streamlit_option_menu import option_menu

# Initialize session state variables for login and chat
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "order_started" not in st.session_state:
    st.session_state.order_started = False
if "user_type" not in st.session_state:
    st.session_state.user_type = None
if "username" not in st.session_state:
    st.session_state.username = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "orders" not in st.session_state:
    st.session_state.orders = []
if "order_form_submitted" not in st.session_state:
    st.session_state.order_form_submitted = False

if "otp" not in st.session_state:
    st.session_state.otp = None  # No OTP initially
if "verified" not in st.session_state:
    st.session_state.verified = False
if "email" not in st.session_state:
    st.session_state.email = ""

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

@st.cache_data
def load_data():
    return get_product_names()

# Function to generate OTP
def generateOTP():
    return str(random.randint(100000, 999999))

# OTP Verification Dialog
@st.dialog("Verify Your Mail")
def verify_mail():
    if st.session_state.otp is None:
        st.session_state.otp = generateOTP()  # Generate OTP only once
        OTP_verification_email(st.session_state.otp,st.session_state.email)
    
    st.write(f"📩 Check your inbox to verify your email and continue registration {st.session_state.email}")
    st.write(f"OTP: {st.session_state.otp}")  # Show OTP for testing (remove in production)
    
    user_otp = st.text_input("Enter OTP")
    left , right = st.columns(2)

    with left:
        if st.button("Submit",use_container_width=True):
            if user_otp == st.session_state.otp:
                st.write("✅ Mail verified successfully!")
                st.session_state.verified = True
                
                st.rerun()
            else:
                st.write("❌ Invalid OTP. Please check your mail again.")
    
    with right:
        if st.button("Resend OTP",use_container_width=True):
            st.session_state.otp = generateOTP()  # Regenerate OTP
            st.write("🔄 OTP has been resent!")

# Login & Registration Page
def login_page():
    st.title("🏥 Healthcare E-Commerce Platform")
    st.subheader("", divider="rainbow")
    
    tab1, tab2 = st.tabs(["LOGIN", "REGISTER"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password",type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if login_user(username, password) == 1:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    user_cred = get_user_details(username)
                    st.session_state.user_cred = user_cred
                    st.session_state.user_type = user_cred["usertype"]
                    st.session_state.email = user_cred["mail"]
                    st.success(f"Successfully logged in as {st.session_state.user_type}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        if not st.session_state.verified:
            st.write("🔐Please verify your email to complete the registration process.")
            with st.form("Verify Email"):
                email = st.text_input("Email")
                verify = st.form_submit_button("Verify")
                
                if verify:
                    st.session_state.email = email
                    verify_mail()

        if st.session_state.verified:
            st.write(f"✅ Email verified successfully! - {st.session_state.email}")
            with st.form("register_form"):
                name = st.text_input("Name")
                username = st.text_input("Username")
                # email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                email = st.session_state.email
                user_type = st.selectbox("Select User Type", ["Doctor", "Patient"])
                region = st.selectbox("Select Region", ["India", "USA", "UK", "Canada", "Australia"])

                register = st.form_submit_button("Register")
                
                if register:
                    user_cred = get_user_details(username)
                    st.session_state.user_cred = user_cred
                    msg = register_user(name, username, password, email, user_type, region)
                    if msg["status"] == "success":
                        st.success(msg["message"])
                        user_cred = get_user_details(username)
                        st.session_state.user_cred = user_cred
                    else:
                        st.error(msg["message"])


def order():
    # 1. as sidebar menu

# 2. horizontal menu
    selected = option_menu(
        menu_title=None,  # required
        options=["New Order", "Order History"],  # required
        icons=["cart4", "hourglass-split"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
    )

    if selected == "Order History":
        # pass
        # st.write("Order History")
        past_orders = get_user_orders(st.session_state.username)
        if len(past_orders)>0:
            # Expandable sections for detailed order view
            for i, order in enumerate(past_orders):
                with st.expander(f"🛒 Order {i+1}: {order['Product']} ({order['Timestamp']})"):
                    st.write(f"**🆔 Order ID:** {order['OrderID']}")
                    st.write(f"**📧 Email:** {order['Email']}")
                    st.write(f"**📦 Product:** {order['Product']}")
                    st.write(f"**🔢 Quantity:** {order['Quantity']}")
                    st.write(f"**📍 Address:** {order['Address']}")
                    st.write(f"**💳 Payment Method:** {order['PaymentMethod']}")
                    st.write(f"**⏰ Timestamp:** {order['Timestamp']}")
        else:
            st.warning("No orders found.")


    if selected == "New Order":

        if st.session_state.order_form_submitted:
            st.success("Your order has been placed successfully!")
            past_orders = get_user_orders(st.session_state.username)
            order_details = past_orders[-1]

            # Display order details with markdown
            st.markdown("## 🛒 Order Details")
            st.write(f"**🆔 Order ID:** {order_details['OrderID']}")
            st.write(f"**📧 Email:** {order_details['Email']}")
            st.write(f"**📦 Product:** {order_details['Product']}")
            st.write(f"**🔢 Quantity:** {order_details['Quantity']}")
            st.write(f"**📍 Address:** {order_details['Address']}")
            st.write(f"**💳 Payment Method:** {order_details['PaymentMethod']}")
            st.write(f"**⏰ Timestamp:** {order_details['Timestamp']}")
            st.markdown("---")

            left , right = st.columns(2, vertical_alignment="bottom")

            if left.button("Place Another Order",use_container_width=True):
                st.session_state.order_started = True
                st.session_state.order_form_submitted = False
                st.rerun()
            elif right.button("Return to Chat",use_container_width=True):
                st.session_state.order_started = False
                st.session_state.order_form_submitted = False
                st.rerun()

            send_order_email(order_details,st.session_state.email)

        else:
            # with st.form("order_form",clear_on_submit=False):
            st.markdown("---")
            st.subheader("📦 Place Your Order")
            # st.markdown("---")
            product = st.selectbox("🔍 Select a product:", load_data())
            # image_url = "./assets/herbal_tea.png"

            col1, col2, col3 = st.columns([1.25, 1, 1.25])
            with col2:
                # st.image(image_url)
                image_data = get_product_image(product)
                if image_data:
                    st.image(image_data, caption=product)
                else:
                    st.warning("⚠️ No image found for this product.")
                
            quantity = st.number_input("Enter quantity:", min_value=1, step=1)
            address = st.text_area("Enter delivery address:")
            email = st.text_input("Enter email for confirmation:")
            paymentmethod = st.radio("Payment Method:", ["COD", "Credit Card", "UPI"])

            st.markdown("---")
            left , right = st.columns(2, vertical_alignment="bottom")

            if left.button("Confirm Order",use_container_width=True):
                st.session_state.order_form_submitted = True
                
                order_details = {
                    "OrderID": str(uuid.uuid4()),
                    "Email": email,
                    "Product": product,
                    "Quantity": quantity, 
                    "Address": address,
                    "PaymentMethod": paymentmethod,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                # st.session_state.orders.append(order_details)
                place_order(st.session_state.username, order_details)
                st.session_state.messages.append({"role": "assistant", "content": f"✅ Your order for 🛒 '**{product}**' has been placed successfully! 🎉"})
                st.success("Order placed successfully!")
                st.rerun()


            elif right.button("Return to Chat",use_container_width=True):
                    st.session_state.order_started = False
                    st.session_state.order_form_submitted = False
                    st.rerun()


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

    welcome_text = """This AI-powered chatbot is designed to enhance your healthcare e-commerce experience by providing you with accurate **product information** 📋, seamless 💰 **price comparison**, and effortless 🛒 **ordering** for your practice and patients. It has specialized features for medical professionals."""

    st.markdown(welcome_text)
    st.subheader("", divider="rainbow")

    # Main chatbot functionality
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")

    
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

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
        template ="""You are a healthcare e-commerce assistant that provides factual, direct answers based solely on the provided context. 

        IMPORTANT: Do not add greetings, introductions, or closing questions when responding to direct queries. Only respond with relevant information from the context.

        RULES:
        - If the user's message is a greeting (like "hi", "hello", "hey","how are u" etc.) or contains only small talk, respond with a friendly greeting
        - Answer directly without adding "Hi there" or "I'm happy to help" introductions
        - Do not ask follow-up questions like "Do you have any other questions?"
        - Only acknowledge greetings if the user's message is purely a greeting with no question
        - Use simple, patient-friendly language while being factual
        - Only use information found in the context
        - Say "I don't have enough information to answer that" if the context doesn't contain relevant information

        Context:
        {context}
        
        Patient's Question:
        {question}
        """


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
            # validation = validator_agent(query)

            # if validation == "YES":

                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)
                
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
                        st.session_state.order_started = True
                        st.session_state.order_form_submitted = False
                        st.rerun()
                    else:
                        # Patient tried to order - notify that ordering is restricted
                        restricted_msg = "🚫 Oops! Ordering products is not applixcable for **patients**.If you need medications 💊 or healthcare products, please consult with your healthcare provider.👩‍⚕️✅"
                        with st.chat_message("assistant"):
                            st.markdown(restricted_msg)
                        st.session_state.messages.append({"role": "assistant", "content": restricted_msg})
                
                elif routed_agent["intent"] == "SUMMARY":
                    # Check user type for order permission
                    start_time = time.time()
                    summary = summary_agent(query, llm, vectordb)
                    end_time = time.time()
                    time_taken = end_time - start_time
                    response_with_time = f"{summary}\n\n*(Response generated in {time_taken:.2f} seconds)*"
                    with st.chat_message("assistant"):
                        st.markdown(response_with_time)
                        st.session_state.messages.append({"role": "assistant", "content": response_with_time})

                        
                else:
                    # Handle other intents
                    response = f"Query will be routed to the {routed_agent['intent']} agent..."
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
def main():
    if not st.session_state.logged_in:
        login_page()
    elif st.session_state.order_started:
        order()
    else:
        chatbot_interface()
    
    # Debug section
    with st.sidebar.expander("Debug: Session State"):
        st.write(st.session_state)

if __name__ == "__main__":
    main()

