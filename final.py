import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_chroma import Chroma
from dotenv import load_dotenv
import random
import json
import time
import uuid
from datetime import datetime
from agents.router_agent import router_agent
from agents.summarization_agent import summary_agent 
from agents.info_agent import response_generator
from agents.recommender_agent import ask_health_bot
from agents.orderagent_recommendation import recommend_similar_products
from src.database import save_feedback, chat_summary, get_product_image, get_product_names
from mainfunction import hybrid_recommendation_tool
from comparison_new import final_result
from agents.send_email import send_order_email,OTP_verification_email
from mongodb import register_user, login_user, get_user_details, place_order, get_user_orders
from streamlit_option_menu import option_menu
import os

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
from langchain_groq import ChatGroq
llm = ChatGroq(model_name="llama-3.3-70b-versatile")
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# OTP Verification Dialog
@st.dialog("Verify Your Mail")
def verify_mail():
    if st.session_state.otp is None:
        st.session_state.otp = generateOTP()  # Generate OTP only once
        OTP_verification_email(st.session_state.email, st.session_state.otp)
    
    st.write(f"📩 Check your inbox to verify your email and continue registration {st.session_state.email}")
    # st.write(f"OTP: {st.session_state.otp}")  # Show OTP for testing (remove in production)
    
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
    st.title("🏥 MediMate: Care Meets Convenience.")
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


@st.dialog("recommendation", width="large")
def show_recom_prod(query):
    response = hybrid_recommendation_tool("saloni", query)
    # st.markdown(recommendations)
    # st.markdown("## 🧾 Recommended Products")
    num_cols = 3  # Number of columns for layout
    rows = [response[i:i + num_cols] for i in range(0, len(response), num_cols)]

    for row in rows:
        cols = st.columns(len(row))
        for idx, product in enumerate(row):
            with cols[idx]:
                st.image(get_product_image(product['name']), use_container_width='always')
                st.markdown(f"### 🧴 {product['name']}")
                st.markdown(f"📄 **Description:** {product['description'][:100]}...")
                
                if product.get("Contents"):
                    st.markdown(f"🧪 **Contents:** {product['Contents']}")
                
                if product.get("benefits"):
                    st.markdown(f"💡 **Benefits:** {product['benefits']}")

                if product.get("adjusted_score"):
                    st.markdown(f"📊 **Score:** {product['adjusted_score']:.2f}")
                
                if product.get("pdf_link"):
                    st.markdown(f"[📘 View PDF]({product['pdf_link']})", unsafe_allow_html=True)

                st.markdown("---")

@st.fragment(run_every=1.0)  # Check every second
def check_inactivity():
    if len(st.session_state.messages) > 0:  # Only start checking after first message
        current_time = time.time()
        inactive_time = current_time - st.session_state.last_activity_time
        
        # Show analysis if inactive for 10 seconds and analysis not shown yet
        if inactive_time >= 120 and not st.session_state.analysis_shown:
            analysis = "⚠️ You will be logged out in 2 seconds due to inactivity"
            
            # Store the analysis in session state
            st.session_state.warning = analysis
            st.session_state.warning_shown = True
        
        # Display the analysis if it exists
        if st.session_state.get("warning_shown", False):
            with st.container():
                st.warning(st.session_state.get("warning", ""), icon="⚠️")

        # Perform logout after 60 seconds of inactivity
        if inactive_time >= 130:
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

            st.markdown("## 🛒 Order Details")
            st.write(f"**🆔 Order ID:** {order_details['OrderID']}")
            st.write(f"**📧 Email:** {order_details['Email']}")
            st.write(f"**📦 Product:** {order_details['Product']}")
            st.write(f"**🔢 Quantity:** {order_details['Quantity']}")
            st.write(f"**📍 Address:** {order_details['Address']}")
            st.write(f"**💳 Payment Method:** {order_details['PaymentMethod']}")
            st.write(f"**⏰ Timestamp:** {order_details['Timestamp']}")
            st.markdown("---")

            st.subheader("✨ You might also like these products! 🛍️")

            with st.spinner("🔍 Finding similar products..."):
                rec = recommend_similar_products(order_details['Product'])
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
                    "OrderID": str(uuid.uuid4()),
                    "Email": email,
                    "Product": product,
                    "Quantity": quantity, 
                    "Address": address,
                    "PaymentMethod": paymentmethod,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    st.title("🤖 MediMate: Care Meets Convenience.")
    
    with st.sidebar:
        st.success(f"Logged in as: {st.session_state.username} ({st.session_state.user_type})")
        a = st.toggle("Use chat history for analysis" )
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

                st.rerun()
            
            elif routed_agent["intent"] == "INFO":
                response = response_generator(query)
                end_time = time.time()
                time_taken = end_time - start_time
                response_with_time = f"{response}\n\n*(Response generated in {time_taken:.2f} seconds)*"
                st.session_state.messages.append({"role": "assistant", "content": response_with_time})

                st.rerun()
                
            elif routed_agent["intent"] == "ORDER":
                if st.session_state.user_type == "Doctor":
                    st.session_state.order_started = True
                    st.session_state.order_form_submitted = False
                else:
                    restricted_msg = "🚫 Oops! Ordering products is not applicable for patients."
                    st.session_state.messages.append({"role": "assistant", "content": restricted_msg})

                st.rerun()

            elif routed_agent["intent"] == "COMPARE":
                response_raw = final_result(query)

                # Parse JSON if response is a string
                if isinstance(response_raw, str):
                    try:
                        response = json.loads(response_raw)
                    except json.JSONDecodeError:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "❌ Failed to parse comparison result. Please try again."
                        })
                        st.stop()
                else:
                    response = response_raw

                # Now build chat-style markdown response
                chat_response = "**🧾 Comparison Results:**\n"

                for product in response:
                    chat_response += f"\n**🧴 {product['name']}**\n"
                    chat_response += f"💸 Price: {product['price']}\n"
                    chat_response += f"🌟 Rating: {product['rating']}\n"
                    chat_response += f"🛒 Source: {product['source']}\n"
                    chat_response += f"[🔗 View Product]({product['link']})\n"

                    if product.get("details"):
                        chat_response += "📌 Details:\n"
                        for detail in product["details"]:
                            chat_response += f"- {detail}\n"

                    if product.get("certifications"):
                        chat_response += f"✅ Certifications: {', '.join(product['certifications'])}\n"

                    chat_response += "\n---\n"

                # Append to chat
                st.session_state.messages.append({"role": "assistant", "content": chat_response})

                st.rerun()
                
            
            elif routed_agent["intent"] == "SUMMARY":
                summary = summary_agent(query)
                end_time = time.time()
                time_taken = end_time - start_time
                response_with_time = f"{summary}\n\n*(Response generated in {time_taken:.2f} seconds)*"
                st.session_state.messages.append({"role": "assistant", "content": response_with_time})
            
                st.rerun()

            # elif routed_agent["intent"] == "RECOMMEND":
                # show_recom_prod(query)

            elif routed_agent["intent"] == "RECOMMEND":
                st.info("⏳ Processing your query, please wait...")

                recommendations = hybrid_recommendation_tool("saloni", query)

                if not recommendations:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "❌ No recommendations found for your query."
                    })
                else:
                    # Chat response for history
                    chat_response = "**🧾 Recommended Products:**\n"
                    for product in recommendations:
                        chat_response += f"\n**{product['name']}**\n"
                        chat_response += f"📄 Description: {product['description'][:150]}...\n"
                        if product.get("pdf_link"):
                            chat_response += f"[View PDF]({product['pdf_link']})\n"

                    end_time = time.time()
                    time_taken = end_time - start_time
                    chat_response += f"\n\n*(Response generated in {time_taken:.2f} seconds)*"
                    st.session_state.messages.append({"role": "assistant", "content": chat_response})

                    # UI: Render product cards with container
                    st.subheader("🧾 Recommended Products")

                    chunk_size = 3
                    for i in range(0, len(recommendations), chunk_size):
                        with st.container():  # Wrap row in a container
                            row = recommendations[i:i+chunk_size]
                            cols = st.columns(len(row))
                            for col, product in zip(cols, row):
                                with col:
                                    image_data = get_product_image(product['name'])
                                    if image_data:
                                        st.image(image_data, caption=product['name'], use_container_width=True)
                                    else:
                                        st.markdown(f"**{product['name']}** (No image available)")

                                    st.markdown(f"**Description:** {product['description']}...")
                                    if product.get("pdf_link"):
                                        st.markdown(f"[📄 View PDF]({product['pdf_link']})", unsafe_allow_html=True)


            else:
                response = f"Query will be routed to the {routed_agent['intent']} agent..."
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        # st.rerun()

    # Run the inactivity checker
    check_inactivity()

def main():
    if not st.session_state.logged_in:
        login_page()
    elif st.session_state.order_started:
        order()
    else:
        chatbot_interface()

    # Debug section
    # with st.sidebar.expander("Debug: Session State"):
    #     st.write(st.session_state) 

if __name__ == "__main__":
    main()