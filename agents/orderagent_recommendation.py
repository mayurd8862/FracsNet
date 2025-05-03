# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams, PointStruct

# # ✅ Load and clean dataset
# df = pd.read_csv("C:/Users/A/Desktop/Internship_Project/product_data.csv")
# df.columns = df.columns.str.replace(" ", "").str.strip()

# def combine_text(row):
#     return f"{row['ProductName']} - {row['FormulatedFor']}. {row['Description']}"

# df["combined_text"] = df.apply(combine_text, axis=1)

# # ✅ Create embeddings
# model = SentenceTransformer('all-MiniLM-L6-v2')
# df["embedding"] = df["combined_text"].apply(lambda x: model.encode(x))

# # ✅ Qdrant client setup
# qdrant_client = QdrantClient(
#     url="https://b53fcd20-f83a-4aed-884e-902e0282a33e.us-east4-0.gcp.cloud.qdrant.io",
#     api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.dsxhYlJXR2wQupPlbJFvAbjKdZWV42H9lJhcqN91WLs",
#     timeout=60.0
#     # ⛔ Replace with your actual key if it's expired
# )

# collection_name = "products_recommendation"

# # ✅ Use non-deprecated method to create collection
# if not qdrant_client.collection_exists(collection_name):
#     qdrant_client.create_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(size=len(df["embedding"].iloc[0]), distance=Distance.COSINE)
#     )

# # ✅ Prepare points for upsert
# points = [
#     PointStruct(
#         id=i,
#         vector=vector.tolist(),
#         payload={
#             "ProductName": df["ProductName"].iloc[i],
#             "FormulatedFor": df["FormulatedFor"].iloc[i],
#             "Description": df["Description"].iloc[i]
#         }
#     )
#     for i, vector in enumerate(df["embedding"])
# ]

# # ✅ Upload to Qdrant
# try:
#     qdrant_client.upsert(collection_name=collection_name, points=points)
#     print("✅ Embeddings uploaded successfully.")
# except Exception as e:
#     print("❌ Error during upload:", str(e))

# # ✅ Recommendation Function
# def recommend_similar_products(ordered_product_name, top_k=3):
#     product_row = df[df["ProductName"] == ordered_product_name]
#     if product_row.empty:
#         return "Product not found in dataset"

#     query_vector = product_row["embedding"].values[0]

#     try:
#         search_result = qdrant_client.search(
#             collection_name=collection_name,
#             query_vector=query_vector,
#             limit=top_k+1
#         )

#         recommendations = [
#             hit.payload["ProductName"]
#             for hit in search_result
#             if hit.payload["ProductName"] != ordered_product_name
#         ][:top_k]

#         return recommendations
#     except Exception as e:
#         return f"❌ Qdrant search failed: {str(e)}"

# # ✅ Example usage
# ordered = "C-Flav"
# recommended = recommend_similar_products(ordered)
# print(f"Because you ordered '{ordered}', you might also like:")
# for r in recommended:
#     print(" ", r)










import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os 
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ✅ 1. Load and clean dataset
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.replace(" ", "").str.strip()

    def combine_text(row):
        return f"{row['ProductName']} - {row['FormulatedFor']}. {row['Description']}"
    
    df["combined_text"] = df.apply(combine_text, axis=1)
    return df

# ✅ 2. Create sentence embeddings
def generate_embeddings(df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
    model = SentenceTransformer(model_name, device='cpu')  # Force CPU to avoid memory issues
    embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)
    df["embedding"] = embeddings.tolist()  # Convert 2D array to list of lists
    return df

# ✅ 3. Initialize Qdrant client
def get_qdrant_client(api_url: str, api_key: str) -> QdrantClient:
    return QdrantClient(url=api_url, api_key=api_key, timeout=60.0)

# ✅ 4. Create Qdrant collection if not exists
def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

# ✅ 5. Upload data to Qdrant
def upload_embeddings_to_qdrant(client: QdrantClient, df: pd.DataFrame, collection_name: str):
    points = [
        PointStruct(
            id=i,
            vector=embedding,
            payload={
                "ProductName": df["ProductName"].iloc[i],
                "FormulatedFor": df["FormulatedFor"].iloc[i],
                "Description": df["Description"].iloc[i]
            }
        )
        for i, embedding in enumerate(df["embedding"])
    ]
    client.upsert(collection_name=collection_name, points=points)
    print("✅ Embeddings uploaded successfully.")

# ✅ 6. Recommendation function
def recommend_similar_products(
    ordered_product_name: str,
    top_k: int = 3
):

    
    # csv_path = "C:/Users/mayur/Desktop/FRACSNET/knowledge/product_data.csv"
    csv_path = os.path.join(os.getcwd(), "knowledge", "product_data.csv")
    qdrant_url = "https://b53fcd20-f83a-4aed-884e-902e0282a33e.us-east4-0.gcp.cloud.qdrant.io"
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.dsxhYlJXR2wQupPlbJFvAbjKdZWV42H9lJhcqN91WLs"
    collection_name = "products_recommendation"

    df = load_and_prepare_data(csv_path)
    df = generate_embeddings(df)
    client = get_qdrant_client(qdrant_url, api_key)

    product_row = df[df["ProductName"] == ordered_product_name]
    if product_row.empty:
        return "Product not found in dataset"

    query_vector = product_row["embedding"].values[0]

    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k + 1
        )

        recommendations = [
            hit.payload["ProductName"]
            for hit in search_result
            if hit.payload["ProductName"] != ordered_product_name
        ][:top_k]

        return recommendations
    except Exception as e:
        return f"❌ Qdrant search failed: {str(e)}"

# # ✅ 7. Main runner
# def main():
#     ordered = "C-Flav"
#     recommended = recommend_similar_products(ordered)
#     print(recommended)

# # ✅ Run the pipeline
# if __name__ == "__main__":
#     main()

