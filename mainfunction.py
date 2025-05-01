import os
import base64
import pymongo
import pandas as pd
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
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

# Dataset
df = pd.read_csv("./knowledge/Cleaned_Dataset.csv").fillna("")
df["combined_text"] = df[["ProductName", "Nutrient_category", "Description", "Formulated_For", "HealthConcern", "Benefits"]].astype(str).agg(" ".join, axis=1)

# Models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = OllamaLLM(model="gemma:2b", base_url="http://localhost:11434")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Functions ---

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
                   node.Contents AS Contents,
                   node.Benefits AS benefits,
                   node.HealthConcern AS concern,
                   score
        """, top_k=top_k * 3, embedding=query_embedding)

        filtered = []
        for record in results:
            item = dict(record)
            text_blob = " ".join([
                str(item.get("description", "")),
                str(item.get("concern", "")),
                str(item.get("benefits", ""))
            ]).lower()
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
    return final

def display_results(results):
    for idx, item in enumerate(results, start=1):
        print(f"\n🔹 Recommendation {idx}")
        print(f"Product Name: {item['name']}")
        print(f" Description: {item['description']}")
        if item.get("pdf_link"):
            print(f" PDF Link: {item['pdf_link']}")
        if item.get("image"):
            print(f" Image URL/Base64: {item['image']}")

# --- CLI Entry Point ---

if __name__ == "__main__":
    print(" Welcome to the Health Supplement CLI Recommender")
    username = input("👤 Enter your registered username: ").strip()
    user = users_col.find_one({"username": username})
    
    if not user:
        print("User not found. Please register first in the database.")
    else:
        user_query = input("💬 Ask your health-related query: ").strip()
        print("\n⏳ Processing your query, please wait...\n")
        recommendations = hybrid_recommendation_tool(username, user_query)
        display_results(recommendations)
