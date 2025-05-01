# import streamlit as st
# import requests

# # Backend URL
# API_URL = "http://localhost:8000"

# st.title("💊 Healthcare E-Commerce Chatbot")

# # Sidebar navigation
# menu = st.sidebar.radio("Navigation", ["Register", "Login", "User Orders", "Place Order", "Ask AI", "Submit Feedback"])

# # Session state for user authentication
# if "username" not in st.session_state:
#     st.session_state.username = None


# # 📌 User Registration
# if menu == "Register":
#     st.subheader("🔹 Register New User")
#     name = st.text_input("Full Name")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     email = st.text_input("Email")
#     usertype = st.selectbox("User Type", ["Customer", "Admin"])
#     region = st.text_input("Region")

#     if st.button("Register"):
#         data = {
#             "name": name, "username": username, "password": password,
#             "mail": email, "usertype": usertype, "region": region
#         }
#         res = requests.post(f"{API_URL}/register", json=data)
#         if res.status_code == 200:
#             st.success("✅ Registration successful!")
#         else:
#             st.error(f"❌ {res.json()['detail']}")


# # 📌 User Login
# elif menu == "Login":
#     st.subheader("🔹 User Login")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")

#     if st.button("Login"):
#         res = requests.post(f"{API_URL}/login/", json={"username": username, "password": password})
#         if res.status_code == 200:
#             st.session_state.username = username
#             st.success("✅ Login successful!")
#         else:
#             st.error(f"❌ {res.json()['detail']}")


# # 📌 Fetch User Orders
# elif menu == "User Orders":
#     if st.session_state.username:
#         st.subheader(f"🛒 Orders of {st.session_state.username}")
#         res = requests.get(f"{API_URL}/users/{st.session_state.username}/orders")
#         if res.status_code == 200:
#             st.write(res.json())
#         else:
#             st.error("❌ No orders found.")
#     else:
#         st.warning("⚠️ Please login first.")


# # 📌 Place Order
# elif menu == "Place Order":
#     if st.session_state.username:
#         st.subheader("🛍️ Place a New Order")
#         email = st.text_input("Email")
#         product = st.text_input("Product Name")
#         quantity = st.number_input("Quantity", min_value=1, step=1)
#         address = st.text_area("Delivery Address")
#         payment = st.selectbox("Payment Method", ["Credit Card", "UPI", "COD"])

#         if st.button("Place Order"):
#             data = {
#                 "email": email, "product": product,
#                 "quantity": quantity, "address": address, "paymentmethod": payment
#             }
#             res = requests.post(f"{API_URL}/users/{st.session_state.username}/orders", json=data)
#             if res.status_code == 200:
#                 st.success("✅ Order placed successfully!")
#             else:
#                 st.error(f"❌ {res.json()['detail']}")
#     else:
#         st.warning("⚠️ Please login first.")


# # 📌 Ask AI Query
# elif menu == "Ask AI":
#     st.subheader("🤖 Ask AI a Question")
#     query = st.text_area("Enter your question:")

#     if st.button("Get Answer"):
#         res = requests.post(f"{API_URL}/ask", json={"question": query})
#         if res.status_code == 200:
#             st.write("**Answer:**", res.json()["answer"])
#         else:
#             st.error("❌ Unable to fetch answer.")


# # 📌 Feedback Section
# elif menu == "Submit Feedback":
#     st.subheader("📝 Submit Feedback")
#     query = st.text_input("Enter the query you asked:")
#     response = st.text_area("Enter the system's response:")
    
#     feedback = st.radio("Was this response helpful?", ["👍 Yes", "👎 No"], index=None)
#     suggestion = ""

#     if feedback == "👎 No":
#         suggestion = st.text_area("Provide your suggestion:")

#     if st.button("Submit Feedback"):
#         if feedback is None:
#             st.error("Please select 👍 or 👎 before submitting.")
#         elif feedback == "👎 No" and not suggestion:
#             st.error("Suggestion is required when selecting 👎.")
#         else:
#             feedback_data = {
#                 "query": query, "response": response,
#                 "feedback": feedback == "👍 Yes", "suggestion": None if feedback == "👍 Yes" else suggestion
#             }
#             res = requests.post(f"{API_URL}/feedback", json=feedback_data)
#             if res.status_code == 200:
#                 st.success("✅ Feedback submitted successfully!")
#             else:
#                 st.error("❌ Error submitting feedback.")






import streamlit as st
import json
from comparison_new import final_result

st.title("🧴 Vitamin C Product Finder")

query = st.text_input("Enter your query:")

if query:
    results_raw = final_result(query)

    # Convert JSON string to Python object if needed
    if isinstance(results_raw, str):
        try:
            results = json.loads(results_raw)
        except json.JSONDecodeError:
            st.error("Failed to parse product results.")
            st.stop()
    else:
        results = results_raw

    # Display products if format is correct
    if isinstance(results, list) and all(isinstance(product, dict) for product in results):
        for product in results:
            with st.container():
                st.markdown(f"""#### 🧴 {product['name']}
**💸 Price:** {product['price']}  
**🌟 Rating:** {product['rating']}  
**🛒 Source:** {product['source']}  
[🔗 View Product]({product['link']})  
""")
                st.markdown("**📌 Details:**")
                for item in product['details']:
                    st.markdown(f"- {item}")

                if product.get("certifications"):
                    st.markdown(f"✅ **Certifications:** {', '.join(product['certifications'])}")

                st.markdown("---")
    else:
        st.error("The results are not in the expected format.")
