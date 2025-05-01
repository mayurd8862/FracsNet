import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
import random
import time
import uuid
from datetime import datetime
from agents.router_agent import router_agent
from agents.summarization_agent import summary_agent 
from agents.recommender_agent import ask_health_bot
from agents.orderagent_recommendation import recommend_similar_products
from agents.send_email import send_order_email, OTP_verification_email
from streamlit_option_menu import option_menu
import os
import requests

# Backend API URL
API_URL = "http://localhost:8000"  # Update this if your backend is hosted elsewhere

load_dotenv()

# Initialize session state variables
def initialize_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "user_type" not in st.session_state:
        st.session_state.user_type = None
    if "email" not in st.session_state:
        st.session_state.email = ""
    if "verified" not in st.session_state:
        st.session_state.verified = False
    if "otp" not in st.session_state:
        st.session_state.otp = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    if "order_started" not in st.session_state:
        st.session_state.order_started = False
    if "order_form_submitted" not in st.session_state:
        st.session_state.order_form_submitted = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = True
    if "last_activity_time" not in st.session_state:
        st.session_state.last_activity_time = time.time()
    if "warning_shown" not in st.session_state:
        st.session_state.warning_shown = False
    if "warning" not in st.session_state:
        st.session_state.warning = ""
    if "analysis_shown" not in st.session_state:
        st.session_state.analysis_shown = False

initialize_session_state()

# Initialize LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile")
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Database functions using FastAPI endpoints
def register_user(name, username, password, email, user_type, region):
    data = {
        "name": name, "username": username, "password": password,
        "mail": email, "usertype": user_type, "region": region
    }
    response = requests.post(f"{API_URL}/register", json=data)
    return response.json()

def login_user(username, password):
    response = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
    if response.status_code == 200:
        return 1
    return 0

def get_user_details(username):
    response = requests.get(f"{API_URL}/users/{username}")
    if response.status_code == 200:
        return response.json()
    return None

def place_order(username, order_details):
    response = requests.post(
        f"{API_URL}/users/{username}/orders",
        json=order_details
    )
    return response.json()

def get_user_orders(username):
    response = requests.get(f"{API_URL}/users/{username}/orders")
    if response.status_code == 200:
        return response.json()
    return []

def save_feedback(username, query, response, reason):
    feedback_data = {
        "query": query,
        "response": response,
        "feedback": False,  # Since they're providing negative feedback
        "suggestion": reason
    }
    response = requests.post(f"{API_URL}/feedback", json=feedback_data)
    return response.json()

def chat_summary(username, summary):
    # This endpoint would need to be created in your FastAPI backend
    response = requests.post(
        f"{API_URL}/users/{username}/chat_summary",
        json={"summary": summary}
    )
    return response.json()

def get_product_names():
    # This endpoint would need to be created in your FastAPI backend
    response = requests.get(f"{API_URL}/products")
    if response.status_code == 200:
        return [product["name"] for product in response.json()]
    return ["Product 1", "Product 2", "Product 3"]  # Fallback

def get_product_image(product_name):
    # This endpoint would need to be created in your FastAPI backend
    response = requests.get(f"{API_URL}/products/{product_name}/image")
    if response.status_code == 200:
        return response.content
    return None

# OTP Verification Dialog
@st.dialog("Verify Your Mail")
def verify_mail():
    if st.session_state.otp is None:
        st.session_state.otp = generateOTP()  # Generate OTP only once
        OTP_verification_email(st.session_state.otp, st.session_state.email)
    
    st.write(f"📩 Check your inbox to verify your email and continue registration {st.session_state.email}")
    st.write(f"OTP: {st.session_state.otp}")  # Show OTP for testing (remove in production)
    
    user_otp = st.text_input("Enter OTP")
    left, right = st.columns(2)

    with left:
        if st.button("Submit", use_container_width=True):
            if user_otp == st.session_state.otp:
                st.write("✅ Mail verified successfully!")
                st.session_state.verified = True
                st.rerun()
            else:
                st.write("❌ Invalid OTP. Please check your mail again.")
    
    with right:
        if st.button("Resend OTP", use_container_width=True):
            st.session_state.otp = generateOTP()  # Regenerate OTP
            st.write("🔄 OTP has been resent!")

def generateOTP():
    return str(random.randint(100000, 999999))

# Login & Registration Page
def login_page():
    st.title("🏥 Healthcare E-Commerce Platform")
    st.subheader("", divider="rainbow")
    
    tab1, tab2 = st.tabs(["LOGIN", "REGISTER"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
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

def logout():
    """Function to handle user logout."""
    # Save chat summary if enabled
    if st.session_state.chat_history and len(st.session_state.messages) > 3:
        chat_summary(st.session_state.username, analyze_chat_history())
    
    # Clear session state
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.user_type = None
    st.session_state.messages = []
    st.session_state.order_started = False
    st.session_state.order_form_submitted = False
    st.session_state.analysis_shown = False
    st.session_state.warning_shown = False
    
    # Rerun the app to show login page
    st.rerun()

def analyze_chat_history():
    if "messages" not in st.session_state or len(st.session_state.messages) < 2:
        return "No sufficient chat data to analyze."

    # Format chat messages for LLM
    chat_text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
    )

    # Prompt for LLM
    messages = [
        ("system", "You are an AI assistant that analyzes chatbot conversations. Your task is to: "
                   "Identify any problems the user is facing. Dont provide any extra information."),
        ("human", f"Here is the conversation:\n\n{chat_text}\n\nAnalyze this and summarize."),
    ]

    # Invoke LLM
    ai_response = llm.invoke(messages)
    
    return ai_response.content

@st.dialog("feedback")
def feedback(index, query, res):
    dialog_key = f"show_dialog_{index}"  # Define dialog_key here
    
    reason = st.text_area(f"🤔Please tell us why this response wasn't helpful")
    if st.button("Submit feedback", use_container_width=True, key=f"submit_feedback_{index}"):
        st.write("✅ Feedback submitted successfully")
        st.session_state[dialog_key] = False
        save_feedback(st.session_state.username, query, res, reason.strip())

@st.fragment(run_every=1.0)  # Check every second
def check_inactivity():
    if len(st.session_state.messages) > 0:  # Only start checking after first message
        current_time = time.time()
        inactive_time = current_time - st.session_state.last_activity_time
        
        # Show analysis if inactive for 10 seconds and analysis not shown yet
        if inactive_time >= 50 and not st.session_state.analysis_shown:
            analysis = "⚠️ You will be logged out in 10 seconds due to inactivity"
            
            # Store the analysis in session state
            st.session_state.warning = analysis
            st.session_state.warning_shown = True
        
        # Display the analysis if it exists
        if st.session_state.get("warning_shown", False):
            with st.container():
                st.warning(st.session_state.get("warning", ""), icon="⚠️")

        # Perform logout after 60 seconds of inactivity
        if inactive_time >= 60:
            logout()

def order():
    selected = option_menu(
        menu_title=None,
        options=["New Order", "Order History"],
        icons=["cart4", "hourglass-split"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if selected == "Order History":
        past_orders = get_user_orders(st.session_state.username)
        if len(past_orders) > 0:
            for i, order in enumerate(past_orders):
                with st.expander(f"🛒 Order {i+1}: {order['product']} ({order['timestamp']})"):
                    st.write(f"**🆔 Order ID:** {order['order_id']}")
                    st.write(f"**📧 Email:** {order['email']}")
                    st.write(f"**📦 Product:** {order['product']}")
                    st.write(f"**🔢 Quantity:** {order['quantity']}")
                    st.write(f"**📍 Address:** {order['address']}")
                    st.write(f"**💳 Payment Method:** {order['paymentmethod']}")
                    st.write(f"**⏰ Timestamp:** {order['timestamp']}")
        else:
            st.warning("No orders found.")

    if selected == "New Order":
        if st.session_state.order_form_submitted:
            st.success("Your order has been placed successfully!")
            past_orders = get_user_orders(st.session_state.username)
            order_details = past_orders[-1]

            st.markdown("## 🛒 Order Details")
            st.write(f"**🆔 Order ID:** {order_details['order_id']}")
            st.write(f"**📧 Email:** {order_details['email']}")
            st.write(f"**📦 Product:** {order_details['product']}")
            st.write(f"**🔢 Quantity:** {order_details['quantity']}")
            st.write(f"**📍 Address:** {order_details['address']}")
            st.write(f"**💳 Payment Method:** {order_details['paymentmethod']}")
            st.write(f"**⏰ Timestamp:** {order_details['timestamp']}")
            st.markdown("---")

            st.subheader("✨ You might also like these products! 🛍️")

            rec = recommend_similar_products(order_details['product'])
            cols = st.columns(len(rec))  # Create horizontal layout
            for i, product in enumerate(rec):
                with cols[i]:
                    image_data = get_product_image(product)
                    if image_data:
                        st.image(image_data, caption=product, use_container_width=True)
                    else:
                        st.write(f"{product} (no image)")

            st.markdown("---")

            left, right = st.columns(2, vertical_alignment="bottom")

            if left.button("Place Another Order", use_container_width=True):
                st.session_state.order_started = True
                st.session_state.order_form_submitted = False
                st.rerun()
            elif right.button("Return to Chat", use_container_width=True):
                st.session_state.order_started = False
                st.session_state.order_form_submitted = False
                st.rerun()

            send_order_email(order_details, st.session_state.email)

        else:
            st.markdown("---")
            st.subheader("📦 Place Your Order")
            product = st.selectbox("🔍 Select a product:", get_product_names())

            col1, col2, col3 = st.columns([1.25, 1, 1.25])
            with col2:
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
            left, right = st.columns(2, vertical_alignment="bottom")

            if left.button("Confirm Order", use_container_width=True):
                st.session_state.order_form_submitted = True
                
                order_details = {
                    "order_id": str(uuid.uuid4()),
                    "email": email,
                    "product": product,
                    "quantity": quantity, 
                    "address": address,
                    "paymentmethod": paymentmethod,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                place_order(st.session_state.username, order_details)
                st.session_state.messages.append({"role": "assistant", "content": f"✅ Your order for 🛒 '**{product}**' has been placed successfully! 🎉"})
                st.success("Order placed successfully!")
                st.rerun()

            elif right.button("Return to Chat", use_container_width=True):
                st.session_state.order_started = False
                st.session_state.order_form_submitted = False
                st.rerun()


def chatbot_interface():
    st.title("🤖 Healthcare E-Commerce Chatbot")
    
    with st.sidebar:
        st.success(f"Logged in as: {st.session_state.username} ({st.session_state.user_type})")
        a = st.toggle("Use chat history for analysis")
        if a:
            st.session_state.chat_history = True
        else:
            st.session_state.chat_history = False

        if st.button("Logout"):
            logout()

        with st.expander("Debug: Session State"):
            st.write(st.session_state)

    welcome_text = """This AI-powered chatbot is designed to enhance your healthcare e-commerce experience by providing you with accurate **product information** 📋, seamless 💰 **price comparison**, and effortless 🛒 **ordering** for your practice and patients. It has specialized features for medical professionals."""
    st.markdown(welcome_text)
    st.subheader("", divider="rainbow")

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

    # Display chat history - only show feedback for the most recent assistant message
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Only show feedback for the most recent assistant message that has a user query before it
            if (message["role"] == "assistant" and 
                i > 0 and 
                st.session_state.messages[i-1]["role"] == "user" and
                (i == len(st.session_state.messages)-1 or  # It's the last message
                 st.session_state.messages[i+1]["role"] == "user")):  # Or next is a user message
                
                feedback_key = f"feedback_{i}"
                dialog_key = f"show_dialog_{i}"
                user_query = st.session_state.messages[i-1]["content"]
                bot_response = message["content"]

                if feedback_key in st.session_state.feedback:
                    st.write(f"😊Feedback: {st.session_state.feedback[feedback_key]}")
                else:
                    sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
                    selected = st.feedback("thumbs", key=feedback_key)
                    
                    if selected is not None and feedback_key not in st.session_state.feedback:
                        st.session_state.feedback[feedback_key] = sentiment_mapping[selected]
                        
                        if selected == 0:
                            st.session_state[dialog_key] = True
                            st.session_state.last_activity_time = time.time()
                            st.rerun()
                
                if dialog_key in st.session_state and st.session_state[dialog_key]:
                    feedback(i, user_query, bot_response)

    # Handle user input
    placeholder_text = "What medications are you looking for?" if st.session_state.user_type == "Patient" else "Ask about products, pricing, or clinical information..."
    
    if query := st.chat_input(placeholder_text):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.last_activity_time = time.time()
        st.session_state.analysis_shown = False
        
        # Process the query
        with st.spinner("Thinking..."):
            start_time = time.time()
            routed_agent = router_agent(query)

            if routed_agent["intent"] == "NONE":
                response = routed_agent["response"]
                end_time = time.time()
                time_taken = end_time - start_time
                response_with_time = f"{response}\n\n*(Response generated in {time_taken:.2f} seconds)*"
                st.session_state.messages.append({"role": "assistant", "content": response_with_time})
            
            elif routed_agent["intent"] == "INFO":
                response = response_generator(vectordb, query) if vectordb else "System is currently unavailable. Please try again later."
                end_time = time.time()
                time_taken = end_time - start_time
                response_with_time = f"{response}\n\n*(Response generated in {time_taken:.2f} seconds)*"
                st.session_state.messages.append({"role": "assistant", "content": response_with_time})
                
            elif routed_agent["intent"] == "ORDER":
                if st.session_state.user_type == "Doctor":
                    st.session_state.order_started = True
                    st.session_state.order_form_submitted = False
                else:
                    restricted_msg = "🚫 Oops! Ordering products is not applicable for patients."
                    st.session_state.messages.append({"role": "assistant", "content": restricted_msg})
            
            elif routed_agent["intent"] == "SUMMARY":
                summary = summary_agent(query, llm, vectordb)
                end_time = time.time()
                time_taken = end_time - start_time
                response_with_time = f"{summary}\n\n*(Response generated in {time_taken:.2f} seconds)*"
                st.session_state.messages.append({"role": "assistant", "content": response_with_time})
            
            # elif routed_agent["intent"] == "RECOMMEND":
            #     products = ask_health_bot("saloni", query)
            #         # Create columns
            #     cols = st.columns(len(products))

            #     # Loop through each column and fill it with product info
            #     for idx, col in enumerate(cols):
            #         with col:
            #             st.image(products[idx]['image'], use_column_width=True)
            #             st.subheader(products[idx]['name'])
            #             st.markdown(f"**Description:** {products[idx]['description']}")
            #             st.markdown(f"**Benefits:** {products[idx]['benefits']}")
            #             st.markdown(f"**Score:** {round(products[idx]['adjusted_score'], 2)}")
            #             st.markdown(f"[📄 View PDF]({products[idx]['pdf_link']})", unsafe_allow_html=True)

            else:
                response = f"Query will be routed to the {routed_agent['intent']} agent..."
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()

    # Run the inactivity checker
    check_inactivity()

def main():
    if not st.session_state.logged_in:
        login_page()
    elif st.session_state.order_started:
        order()
    else:
        chatbot_interface()

if __name__ == "__main__":
    main()