# from typing import List
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama
# from dotenv import load_dotenv

# load_dotenv()

# llm = ChatOllama(
#     model="nemotron-mini",
#     temperature=0,
#     num_predict=256,
# )

# class RecommenderResponse(BaseModel):
#     recommended_questions: List[str] = Field(
#         description="List of three follow-up questions related to symptoms, dosage, side effects, and other relevant aspects."
#     )

# def recommend_query(query: str) -> dict:
#     parser = JsonOutputParser(pydantic_object=RecommenderResponse)
    
#     prompt = PromptTemplate(
#         template="""
#         You are a smart AI assistant for a healthcare e-commerce application.
#         Your task is to generate three relevant follow-up questions that a user might ask based on their query.
#         These questions should focus on symptoms, dosage, side effects, and other important aspects related to the product inquiry.
#         Questions should be of short words.
        
#         {format_instructions}
#         Query: {query}
        
#         phrase questions naturally from the chatbot's perspective.

#         """,
#         input_variables=["query"],
#         partial_variables={"format_instructions": parser.get_format_instructions()}
#     )
    
#     chain = prompt | llm | parser
    
#     try:
#         result = chain.invoke({"query": query})
#         return result
#     except Exception as e:
#         print(f"Error in generating recommended questions: {e}")
#         return {"recommended_questions": []}


# def recommend_product(query: str) -> dict:
#     pass
# #############################################
# #        Data science Team working          #
# #############################################




# Core health agent functions (no Streamlit UI)
import pymongo
import pandas as pd
import base64
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from neo4j import GraphDatabase
from PIL import Image
from io import BytesIO
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_community import GoogleSearchAPIWrapper

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
df["combined_text"] = df[["ProductName", "Nutrient_category", "Description", "Formulated_For", "HealthConcern", "Benefits"]].astype(str).agg(" ".join, axis=1)

# Models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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
    query_text = f"Health condition: {user_query}. Age: 19, Gender: female. Looking for effective supplements or natural remedies."
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

# LangChain Agent Tool
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

# Direct usage function
def ask_health_bot(username, user_query):
    product = match_product_by_name(user_query)
    if product:
        product_info = df[df["ProductName"] == product].iloc[0].to_dict()
        record_interaction(username, product)
        record_chat(username, user_query, f"Matched product: {product}")
        return {"type": "matched_product", "product": product_info}
    else:
        response = agent.invoke(f"{username}|{user_query}")
        results = hybrid_recommendation_tool(username, user_query)
        record_chat(username, user_query, response)
        return {"type": "recommendations", "results": results}
