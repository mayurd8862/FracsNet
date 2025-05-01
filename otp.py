


import streamlit as st
import random
from mongodb import register_user, login_user, get_user_details

# Function to generate OTP
def generateOTP():
    return str(random.randint(100000, 999999))  # 6-digit OTP

# Initialize session state for OTP and email verification if not already set
if "otp" not in st.session_state:
    st.session_state.otp = None  # No OTP initially
if "verified" not in st.session_state:
    st.session_state.verified = False
if "email" not in st.session_state:
    st.session_state.email = ""

# OTP Verification Dialog
@st.dialog("Verify Your Mail")
def verify_mail():
    if st.session_state.otp is None:
        st.session_state.otp = generateOTP()  # Generate OTP only once
    
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
                # email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                email = st.session_state.email
                user_type = st.selectbox("Select User Type", ["Doctor", "Patient"])
                region = st.selectbox("Select Region", ["India", "USA", "UK", "Canada", "Australia"])
                register = st.form_submit_button("Register")
                
                if register:
                    msg = register_user(name, username, password, email, user_type, region)
                    if msg["status"] == "success":
                        st.success(msg["message"])
                    else:
                        st.error(msg["message"])

if __name__ == "__main__":
    login_page()
