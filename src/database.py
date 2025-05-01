import pymongo
import bcrypt
import jwt
from datetime import datetime
import streamlit as st
import base64
import uuid
from PIL import Image
from io import BytesIO
# MongoDB Connection
# client = pymongo.MongoClient("mongodb://localhost:27017/")
client = pymongo.MongoClient("mongodb+srv://mayurr:12345@cluster0.hllwy4r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client["fracsnet"]
users_collection = db["user_credentials"]
orders_collection = db["order_details"]
products_collection = db["products"]
feedback_collection = db["feedback"]
chat_summary_collection = db["chat_summary"]

SECRET_KEY = "@mayur123"


def register_user(name, username, password, mail, usertype, region):
        # Check if username already exists
        if users_collection.find_one({"username": username}):
            return "Username already exists"

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Store user in database
        user_data = {
            "name": name,
            "username": username,
            "password": hashed_password.decode('utf-8'),  # Store as string
            "usertype": usertype,
            "mail": mail,
            "region": region,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Insert the user data into the collection
        result = users_collection.insert_one(user_data)
        
        if result.inserted_id:
            return {"success": True, "messege": "User registered successfully"}
        else:
            return {"error": "Failed to register user"}

def login_user(username, password):
    user = users_collection.find_one({"username": username})
    
    if not user:
        return {"error": "User not found"}

    # Verify password
    if bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
        token = jwt.encode(
            {"username": username, "exp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            SECRET_KEY,
            algorithm="HS256"
        )
        return {"success": True, "token": token}  # Return success response with JWT
    else:
        return {"error": "Invalid password"}  # Return dictionary instead of 0


# def get_user_details(username):
#     user = users_collection.find_one({"username": username}, {"_id": 0, "password": 0})  # Exclude _id & password

#     if user:
#         return user
#     else:
#         return 0

def get_user_details(username):
    # Find user by username, excluding _id and password fields
    user = users_collection.find_one(
        {"username": username}, 
        {"_id": 0, "password": 0}
    )
    
    # Return user if found, otherwise return error
    if user:
        return {"success": True, "user": user}
    else:
        return {"error": "User not found"}


def place_order(username, email, product, quantity, address, paymentmethod):
    """
    Store order details for a user. First check if user exists in users_collection.
    If user doesn't exist, don't place the order.
    """
    # First check if user exists in users database
    user_exists = users_collection.find_one({"username": username})
    
    if not user_exists:
        return {"error": "User does not exist. Cannot place order."}
    
    new_order = {
        "OrderID": str(uuid.uuid4()),
        "Email": email,
        "Product": product,
        "Quantity": quantity, 
        "Address": address,
        "PaymentMethod": paymentmethod,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    existing_user_orders = orders_collection.find_one({"name": username})

    if existing_user_orders:
        # Append new order to existing order details
        result = orders_collection.update_one(
            {"name": username},
            {"$push": {"order_details": new_order}}  # Append new order to the list
        )
        if result.modified_count > 0:
            return {"success": True, "message": "Order placed successfully"}
        else:
            return {"error": "Failed to place order"}
    else:
        # Insert new order entry
        order_data = {"name": username, "order_details": [new_order]}  # Store as a list
        result = orders_collection.insert_one(order_data)
        if result.inserted_id:
            return {"success": True, "message": "Order placed successfully"}
        else:
            return {"error": "Failed to place order"}
    

def get_user_orders(username):
    """
    Retrieve all orders placed by a specific user.
    """
    user_orders = orders_collection.find_one({"name": username}, {"_id": 0})  # Exclude MongoDB _id field

    if user_orders:
        return {"success": True, "orders": user_orders["order_details"]}
    else:
        return {"error": "No orders found for this user"}
    
def get_product_names():
    """Fetch all product names from MongoDB and cache them."""
    return [doc["product_name"] for doc in products_collection.find({}, {"product_name": 1})]

# def get_product_image(product_name):
#     """Fetch the base64-encoded image from MongoDB and decode it."""
#     product_data = products_collection.find_one({"product_name": product_name}, {"image_base64": 1})
    
#     if product_data and "image_base64" in product_data:
#         return base64.b64decode(product_data["image_base64"])
#     return None


def get_product_image(product_name):
    """Fetch the base64-encoded image from MongoDB and decode it with fixed size."""
    product_data = products_collection.find_one({"product_name": product_name}, {"image_base64": 1})
    
    if product_data and "image_base64" in product_data:
        try:
            image_bytes = base64.b64decode(product_data["image_base64"])
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            img = img.resize((250, 200))  # Resize to fixed dimensions
            return img
        except Exception as e:
            print(f"Error decoding or resizing image for {product_name}: {e}")
            return None
    return None


def save_feedback(username, query, response, reason):
    """
    Store feedback details for a user. First, check if the user exists in users_collection.
    If the user doesn't exist, don't save the feedback.
    """
    # Check if user exists
    user_exists = users_collection.find_one({"username": username})
    
    if not user_exists:
        return {"error": "User does not exist. Cannot submit feedback."}

    # Create new feedback entry
    new_feedback = {
        "Query": query,
        "Response": response,
        "Reason": reason,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    existing_user_feedback = feedback_collection.find_one({"username": username})

    if existing_user_feedback:
        # Append new feedback to existing user entry
        result = feedback_collection.update_one(
            {"username": username},
            {"$push": {"feedback_details": new_feedback}}  # Append feedback to list
        )
        if result.modified_count > 0:
            return {"success": True, "message": "Feedback submitted successfully"}
        else:
            return {"error": "Failed to submit feedback"}
    else:
        # Insert new feedback entry for the user
        feedback_data = {"username": username, "feedback_details": [new_feedback]}  # Store as list
        result = feedback_collection.insert_one(feedback_data)
        if result.inserted_id:
            return {"success": True, "message": "Feedback submitted successfully"}
        else:
            return {"error": "Failed to submit feedback"}




def chat_summary(username, summary):

    # Create new feedback entry
    chat_summary = {
        "username": username,
        "summary": summary,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    result = chat_summary_collection.insert_one(chat_summary)

chat_summary("mayur", "This is a test summary")