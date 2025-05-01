import pymongo
import bcrypt
import jwt
from datetime import datetime
import streamlit as st

# MongoDB Connection
# client = pymongo.MongoClient("mongodb://localhost:27017/")
client = pymongo.MongoClient("mongodb+srv://mayurr:12345@cluster0.hllwy4r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["fracsnet"]
users_collection = db["user_credentials"]
orders_collection = db["order_details"]

SECRET_KEY = "@mayur123"


from datetime import datetime
import bcrypt

def register_user(name, username, password, mail, usertype, region):
    try:
        # Check if username already exists
        if users_collection.find_one({"username": username}):
            return {"status": "error", "message": "Username already exists"}

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
            return {"status": "success", "message": "User registered successfully"}
        else:
            return {"status": "error", "message": "Failed to register user"}

    except Exception as e:
        return {"status": "error", "message": "An error occurred", "details": str(e)}



def login_user(username, password):
    user = users_collection.find_one({"username": username})
    
    if not user:
        return {"error": "User not found"}

    # Verify password
    if bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
        # Generate JWT token
        token = jwt.encode(
            {"username": username, "exp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            SECRET_KEY,
            algorithm="HS256"
        )
        return 1
    else:
        return 0


def get_user_details(username):
    user = users_collection.find_one({"username": username}, {"_id": 0, "password": 0})  # Exclude _id & password

    if user:
        return user
    else:
        return 0





def place_order(username, new_order):
    """
    Store order details for a user. If the user already has orders, append the new order.
    """
    existing_user = orders_collection.find_one({"name": username})

    if existing_user:
        # Append new order to existing order details
        result = orders_collection.update_one(
            {"name": username},
            {"$push": {"order_details": new_order}}  # Append new order to the list
        )
        return "Order updated successfully" if result.modified_count > 0 else "Failed to update order"
    else:
        # Insert new order entry if user does not exist
        order_data = {"name": username, "order_details": [new_order]}  # Store as a list
        result = orders_collection.insert_one(order_data)
        return "New order placed successfully" if result.inserted_id else "Failed to place order"

def get_user_orders(username):
    """
    Retrieve all orders placed by a specific user.
    """
    user_orders = orders_collection.find_one({"name": username}, {"_id": 0})  # Exclude MongoDB _id field

    if user_orders:
        return user_orders["order_details"]
    else:
        return "No orders found for this user."