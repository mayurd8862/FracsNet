import streamlit as st
import pandas as pd

# Sample JSON data
data = [
    {
        "username": "mayur",
        "summary": {
            "problems_faced": ["Not able to place order", "payment issues", "vit c product not available"],
            "products_searched": ["Vit C supplement", "4 sight"]
        },
        "Timestamp": "2025-03-29 11:20:04"
    },
    {
        "username": "sam123",
        "summary": {
            "problems_faced": ["Delayed delivery", "Product received was damaged"],
            "products_searched": ["Protein powder", "Fish oil capsules"]
        },
        "Timestamp": "2025-03-28 15:45:30"
    },
    {
        "username": "alex22",
        "summary": {
            "problems_faced": ["App keeps crashing", "Can't add items to cart"],
            "products_searched": ["Multivitamins", "Collagen powder"]
        },
        "Timestamp": "2025-03-27 09:10:15"
    },
    {
        "username": "user1234",
        "summary": {
            "problems_faced": ["Promo code not working", "Order canceled without reason"],
            "products_searched": ["Omega-3 capsules", "Calcium tablets"]
        },
        "Timestamp": "2025-03-26 20:05:50"
    },
    {
        "username": "apex22",
        "summary": {
            "problems_faced": ["Wrong item delivered", "Return process is slow"],
            "products_searched": ["Biotin tablets", "Hair growth supplement"]
        },
        "Timestamp": "2025-03-25 14:22:10"
    }
]

# Convert JSON data into a DataFrame
df = pd.DataFrame([
    {
        "Username": entry["username"],
        "Problems Faced": ", ".join(entry["summary"]["problems_faced"]),
        "Products Searched": ", ".join(entry["summary"]["products_searched"]),
        "Timestamp": entry["Timestamp"]
    }
    for entry in data
])

# Streamlit App
st.title("User Issues Table")

st.write("### User Issues Summary")
st.dataframe(df)  # Display table in Streamlit

