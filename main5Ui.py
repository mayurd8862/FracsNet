# agentic_streamlit_health_app.py
import streamlit as st
import requests
import pymongo
import pandas as pd
import base64
import os
from langchain_groq import ChatGroq
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from PIL import Image
from io import BytesIO
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_community import GoogleSearchAPIWrapper

# Styling
st.set_page_config(page_title="HealthAgent", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    h1, h2, h3, h4 { color: #2C3E50; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5em 1em;
        border: none;
        border-radius: 5px;
    }
    .stTextInput>div>input {
        background-color: #ffffff;
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment
load_dotenv()

# Neo4j setup
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

# MongoDB setup
client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client["fracsnet"]
users_col = db["users"]
products_col = db["products"]
interactions_col = db["interactions"]
chat_history_col = db["chat_history"]

# Dataset
df = pd.read_csv("./knowledge/Cleaned_Dataset.csv").fillna("")
df["combined_text"] = df[["ProductName", "Nutrient_category","Description","Formulated_For","HealthConcern", "Benefits"]].astype(str).agg(" ".join, axis=1)

# Models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# llm = OllamaLLM(model="qwen:4b", base_url="http://localhost:11434")
llm = ChatGroq(model_name="llama-3.3-70b-versatile")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
search = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID")
)

# Recommendation logic
def get_neo4j_recommendations(query_text, top_k=5):
    query_embedding = embedding_model.embed_query(query_text)
    keywords = set(query_text.lower().split())

    with driver.session() as session:
        results = session.run("""
            CALL db.index.vector.queryNodes('product_embeddings', $top_k, $embedding)
            YIELD node, score
            RETURN node.ProductName AS name,
                   node.Description AS description,
                   node.ProductImage AS image,
                   node.PdfLink AS pdf_link,
                   node.Benefits AS benefits,
                   node.HealthConcern AS concern,
                   score
        """, top_k=top_k * 3, embedding=query_embedding)

        filtered = []
        for record in results:
            item = dict(record)
            text_blob = " ".join([str(item.get("description", "")), str(item.get("concern", "")), str(item.get("benefits", ""))]).lower()
            overlap = keywords.intersection(set(text_blob.split()))
            adjusted_score = item["score"] + 0.5 * len(overlap)
            if overlap or item["score"] > 0.75:
                item["adjusted_score"] = adjusted_score
                filtered.append(item)

        return sorted(filtered, key=lambda x: x["adjusted_score"], reverse=True)[:top_k]

def get_user_based_recommendations(username, top_k=3):
    user_interactions = interactions_col.find({"username": username})
    product_names = [ui["product_name"] for ui in user_interactions]
    if not product_names:
        return None
    df_user = df[df["ProductName"].isin(product_names)]
    combined_keywords = " ".join(df_user["combined_text"].tolist())
    return get_neo4j_recommendations(combined_keywords, top_k=top_k)

def hybrid_recommendation_tool(username, user_query):
    user = users_col.find_one({"username": username})
    query_text = f"Health condition: {user_query}. Age: {user['age']}, Gender: {user['gender']}. Looking for effective supplements or natural remedies."
    hybrid = get_neo4j_recommendations(query_text)
    user_cf = get_user_based_recommendations(username)
    if user_cf:
        hybrid.extend([item for item in user_cf if item["name"] not in {c["name"] for c in hybrid}])
    seen = set()
    final = []
    for item in hybrid:
        if item["name"] not in seen:
            seen.add(item["name"])
            final.append(item)
        if len(final) >= 3:
            break
    return final or [search.run(user_query)]

def get_product_image(product_name):
    product = products_col.find_one({"product_name": product_name}, {"image_base64": 1})
    if product and "image_base64" in product:
        try:
            decoded = base64.b64decode(product["image_base64"])
            return Image.open(BytesIO(decoded))
        except:
            return None
    return None

def record_interaction(username, product_name):
    interactions_col.insert_one({"username": username, "product_name": product_name})

def record_chat(username, query, response):
    chat_history_col.insert_one({"username": username, "query": query, "response": response})

def get_user_chat_history(username):
    return list(chat_history_col.find({"username": username}, {"_id": 0}))

def match_product_by_name(user_query):
    query_lower = user_query.lower()
    for product in df["ProductName"]:
        if product.lower() in query_lower:
            return product
    return None

# LangChain Tool
def recommendation_agent_tool(input):
    username, query = input.split("|")
    recs = hybrid_recommendation_tool(username.strip(), query.strip())
    return "\n".join([r["name"] if isinstance(r, dict) else r for r in recs])

tools = [Tool(name="Recommender", func=recommendation_agent_tool, description="Use to get supplement recommendations by passing 'username|query' format")]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,
    handle_parsing_errors=True
)

# ---------------- Streamlit UI ------------------
st.title("Health Recommendation Engine")
page = st.sidebar.radio("Navigation", ["Login", "Register", "Chatbot", "Chat History"])

if page == "Register":
    st.subheader("Create Account")
    name = st.text_input("Full Name")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    usertype = st.selectbox("User Type", ["Patient", "Doctor"])
    age = st.number_input("Age", 1, 100)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    region = st.text_input("Region")

    if st.button("Register"):
        if not all([name, username, email, password, confirm, region]):
            st.error("Please fill out all fields.")
        elif password != confirm:
            st.error("Passwords do not match.")
        elif users_col.find_one({"username": username}):
            st.error("Username already exists.")
        else:
            users_col.insert_one({"name": name, "username": username, "mail": email, "password": password, "usertype": usertype, "age": age, "gender": gender, "region": region})
            st.success("Registration successful. You can now login.")

elif page == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = users_col.find_one({"username": username, "password": password})
        if user:
            st.session_state.username = username
            st.success(f"Welcome back, {user['name']}!")
        else:
            st.error("Invalid credentials")

elif page == "Chatbot":
    if "username" not in st.session_state:
        st.warning("Please login first to access the chatbot.")
    else:
        st.subheader(f"Hi {st.session_state.username}, Ask me anything about your health!")
        user_query = st.text_input("💬 Type your question")

        if user_query:
            with st.spinner("Thinking..."):
                # Match product by name (helper function must be defined above)
                matched_product = match_product_by_name(user_query)

                if matched_product:
                    st.success(f"Found product: {matched_product}")
                    product_info = df[df["ProductName"] == matched_product].iloc[0]
                    st.markdown(f"### ✅ {matched_product}")
                    img = get_product_image(matched_product)
                    if img:
                        st.image(img, width=200)
                    st.markdown(f"{product_info['Description'][:5000]}...")
                    if product_info.get("PdfLink"):
                        st.markdown(f"[📄 PDF Reference]({product_info['PdfLink']})")
                    record_interaction(st.session_state.username, matched_product)
                    record_chat(st.session_state.username, user_query, f"Matched product: {matched_product}")
                else:
                    agent_input = f"{st.session_state.username}|{user_query}"
                    response = agent.run(agent_input)
                    results = hybrid_recommendation_tool(st.session_state.username, user_query)

                    cols = st.columns(3)
                    for i, item in enumerate(results):
                        if isinstance(item, str):
                            st.markdown(f"🔎 Google fallback: {item}")
                            continue
                        with cols[i]:
                            st.markdown(f"### ✅ {item['name']}")
                            img = get_product_image(item["name"])
                            if img:
                                st.image(img, width=200)
                            st.markdown(f"{item['description'][:5000]}...")
                            if item.get("pdf_link"):
                                st.markdown(f"[📄 PDF Reference]({item['pdf_link']})")
                            record_interaction(st.session_state.username, item["name"])

                    record_chat(st.session_state.username, user_query, response)


elif page == "Chat History":
    if "username" not in st.session_state:
        st.warning("Please login first.")
    else:
        st.subheader("📜 Chat History")
        chats = get_user_chat_history(st.session_state.username)
        for c in reversed(chats):
            st.markdown(f"**You:** {c['query']}")
            st.markdown(f"**Bot:** {c['response']}")
            st.markdown("---")




# # Core health agent functions (no Streamlit UI)
# import pymongo
# import pandas as pd
# import base64
# import os
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from neo4j import GraphDatabase
# from PIL import Image
# from io import BytesIO
# from langchain_ollama import OllamaLLM
# from langchain.agents import initialize_agent, Tool
# from langchain.agents.agent_types import AgentType
# from langchain.memory import ConversationBufferMemory
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_community import GoogleSearchAPIWrapper

# # Load environment
# load_dotenv()

# # Neo4j setup
# driver = GraphDatabase.driver(
#     os.getenv("NEO4J_URI"),
#     auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
# )

# # MongoDB setup
# client = pymongo.MongoClient(os.getenv("MONGO_URI"))
# db = client["fracsnet"]
# users_col = db["users"]
# products_col = db["products"]
# interactions_col = db["interactions"]
# chat_history_col = db["chat_history"]

# # Dataset
# df = pd.read_csv("./knowledge/Cleaned_Dataset.csv").fillna("")
# df["combined_text"] = df[["ProductName", "Nutrient_category", "Description", "Formulated_For", "HealthConcern", "Benefits"]].astype(str).agg(" ".join, axis=1)

# # Models
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# llm = ChatGroq(model_name="llama-3.3-70b-versatile")
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# search = GoogleSearchAPIWrapper(
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     google_cse_id=os.getenv("GOOGLE_CSE_ID")
# )

# # Recommendation logic
# def get_neo4j_recommendations(query_text, top_k=5):
#     query_embedding = embedding_model.embed_query(query_text)
#     keywords = set(query_text.lower().split())

#     with driver.session() as session:
#         results = session.run("""
#             CALL db.index.vector.queryNodes('product_embeddings', $top_k, $embedding)
#             YIELD node, score
#             RETURN node.ProductName AS name,
#                    node.Description AS description,
#                    node.ProductImage AS image,
#                    node.PdfLink AS pdf_link,
#                    node.Benefits AS benefits,
#                    node.HealthConcern AS concern,
#                    score
#         """, top_k=top_k * 3, embedding=query_embedding)

#         filtered = []
#         for record in results:
#             item = dict(record)
#             text_blob = " ".join([str(item.get("description", "")), str(item.get("concern", "")), str(item.get("benefits", ""))]).lower()
#             overlap = keywords.intersection(set(text_blob.split()))
#             adjusted_score = item["score"] + 0.5 * len(overlap)
#             if overlap or item["score"] > 0.75:
#                 item["adjusted_score"] = adjusted_score
#                 filtered.append(item)

#         return sorted(filtered, key=lambda x: x["adjusted_score"], reverse=True)[:top_k]

# def get_user_based_recommendations(username, top_k=3):
#     user_interactions = interactions_col.find({"username": username})
#     product_names = [ui["product_name"] for ui in user_interactions]
#     if not product_names:
#         return None
#     df_user = df[df["ProductName"].isin(product_names)]
#     combined_keywords = " ".join(df_user["combined_text"].tolist())
#     return get_neo4j_recommendations(combined_keywords, top_k=top_k)

# def hybrid_recommendation_tool(username, user_query):
#     user = users_col.find_one({"username": username})
#     query_text = f"Health condition: {user_query}. Age: 19, Gender: female. Looking for effective supplements or natural remedies."
#     hybrid = get_neo4j_recommendations(query_text)
#     user_cf = get_user_based_recommendations(username)
#     if user_cf:
#         hybrid.extend([item for item in user_cf if item["name"] not in {c["name"] for c in hybrid}])
#     seen = set()
#     final = []
#     for item in hybrid:
#         if item["name"] not in seen:
#             seen.add(item["name"])
#             final.append(item)
#         if len(final) >= 3:
#             break
#     return final or [search.run(user_query)]

# def get_product_image(product_name):
#     product = products_col.find_one({"product_name": product_name}, {"image_base64": 1})
#     if product and "image_base64" in product:
#         try:
#             decoded = base64.b64decode(product["image_base64"])
#             return Image.open(BytesIO(decoded))
#         except:
#             return None
#     return None

# def record_interaction(username, product_name):
#     interactions_col.insert_one({"username": username, "product_name": product_name})

# def record_chat(username, query, response):
#     chat_history_col.insert_one({"username": username, "query": query, "response": response})

# def get_user_chat_history(username):
#     return list(chat_history_col.find({"username": username}, {"_id": 0}))

# def match_product_by_name(user_query):
#     query_lower = user_query.lower()
#     for product in df["ProductName"]:
#         if product.lower() in query_lower:
#             return product
#     return None

# # LangChain Agent Tool
# def recommendation_agent_tool(input):
#     username, query = input.split("|")
#     recs = hybrid_recommendation_tool(username.strip(), query.strip())
#     return "\n".join([r["name"] if isinstance(r, dict) else r for r in recs])

# tools = [Tool(name="Recommender", func=recommendation_agent_tool, description="Use to get supplement recommendations by passing 'username|query' format")]
# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     memory=memory,
#     verbose=False,
#     handle_parsing_errors=True
# )

# # Direct usage function
# def ask_health_bot(username, user_query):
#     product = match_product_by_name(user_query)
#     if product:
#         product_info = df[df["ProductName"] == product].iloc[0].to_dict()
#         record_interaction(username, product)
#         record_chat(username, user_query, f"Matched product: {product}")
#         return {"type": "matched_product", "product": product_info}
#     else:
#         response = agent.run(f"{username}|{user_query}")
#         results = hybrid_recommendation_tool(username, user_query)
#         record_chat(username, user_query, response)
#         return {"type": "recommendations", "results": results}



# # a = ask_health_bot("saloni", "recommend me a supplement for diabetes")

# # print(a["results"])