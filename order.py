# import streamlit as st
# import pymongo
# import base64
# from PIL import Image
# import io

# # MongoDB connection setup
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["fracsnet"]
# collection = db["products"]

# # Fetch all product names from MongoDB
# product_names = [doc["product_name"] for doc in collection.find({}, {"product_name": 1})]

# # Streamlit UI
# st.title("Product Image Viewer")

# # Dropdown to select product
# selected_product = st.selectbox("Select a Product", product_names)

# # Fetch the corresponding image from MongoDB
# if selected_product:
#     product_data = collection.find_one({"product_name": selected_product})
    
#     if product_data and "image_base64" in product_data:
#         base64_data = product_data["image_base64"]
        
#         # Convert base64 to image
#         image_bytes = base64.b64decode(base64_data)
#         image = Image.open(io.BytesIO(image_bytes))
        
#         # Display image
#         st.image(image, caption=selected_product)
#     else:
#         st.error("Image not found for the selected product.")



# import streamlit as st  
# import os
# import pandas as pd
# from src.database import get_product_image, get_product_names
# import uuid
# from datetime import datetime

# if "orders" not in st.session_state:
#     st.session_state.orders = []

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# # Load the CSV file
# @st.cache_data
# def load_data():
#     product_names = get_product_names()
#     return product_names


# def order():

#     with st.form("order_form",clear_on_submit=False):
#         st.subheader("Place Your Order")
#         product = st.selectbox("Select a product:", load_data())
#         image_url = "./assets/herbal_tea.png"
#         # image_url="https://drive.usercontent.google.com/download?id=1iztp-75j-9bqSG5lo6qf5033fkNnv1jf&export=view&authuser=0"

#         col1, col2, col3 = st.columns([1.25, 1, 1.25])
#         with col2:
#             st.image(get_product_image(product))
            
            
#         quantity = st.number_input("Enter quantity:", min_value=1, step=1)
#         address = st.text_area("Enter delivery address:")
#         email = st.text_input("Enter email for confirmation:")
#         paymentmethod = st.radio("Payment Method:", ["COD", "Credit Card", "UPI"])

#         if st.form_submit_button("Confirm Order"):
#             st.session_state.order_form_submitted = True
            
#             order_details = {
#                 "OrderID": str(uuid.uuid4()),
#                 "Email": email,
#                 "Product": product,
#                 "Quantity": quantity, 
#                 "Address": address,
#                 "PaymentMethod": paymentmethod,
#                 "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             }
            
#             # st.session_state.orders.append(order_details)
#             st.session_state.orders.append(order_details)
#             st.success("Order placed successfully!")


# if __name__ == "__main__":
#     order()
#     with st.sidebar.expander("Debug: Session State"):
#         st.write(st.session_state)






import streamlit as st  
import os
import uuid
from datetime import datetime
from src.database import get_product_image, get_product_names

if "orders" not in st.session_state:
    st.session_state.orders = []

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

@st.cache_data
def load_data():
    return get_product_names()

def order():
    st.subheader("📦 Place Your Order")
    
    # Product selection (updates immediately)
    product = st.selectbox("🔍 Select a product:", load_data())

    # Show product image **outside** the form for instant updates
    image_data = get_product_image(product)
    if image_data:
        st.image(image_data, caption=product)
    else:
        st.warning("⚠️ No image found for this product.")

    with st.form("order_form", clear_on_submit=False):
        quantity = st.number_input("Enter quantity:", min_value=1, step=1)
        address = st.text_area("Enter delivery address:")
        email = st.text_input("Enter email for confirmation:")
        paymentmethod = st.radio("Payment Method:", ["COD", "Credit Card", "UPI"])

        if st.form_submit_button("Confirm Order"):
            order_details = {
                "OrderID": str(uuid.uuid4()),
                "Email": email,
                "Product": product,
                "Quantity": quantity, 
                "Address": address,
                "PaymentMethod": paymentmethod,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            st.session_state.orders.append(order_details)
            st.success("✅ Order placed successfully!")

if __name__ == "__main__":
    order()
    with st.sidebar.expander("🛠 Debug: Session State"):
        st.write(st.session_state)
