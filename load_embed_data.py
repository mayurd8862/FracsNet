from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
import sys

def embd_vectordb(filepath):
    # Initialize the embedding model
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    try:
        # Load and split the PDF document
        docs = PyPDFLoader(filepath).load_and_split()

        # Create a Chroma vector store with a specified directory for persistence
        db = Chroma.from_documents(docs, embedding, persist_directory="./chroma_db")
        print("Vector database created and persisted.")
        return db
    except Exception as e:
        print(f"Error creating vector database: {e}")
        return None

def load_vectordb():
    # Initialize the embedding model
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        # Load the previously saved Chroma vector store
        loaded_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
        return loaded_db
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None

if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     filepath = sys.argv[1]
    #     embd_vectordb(filepath)

    # loaded_db = load_vectordb()

    # if loaded_db:
    #     # Perform retrieval operations
    #     query = "What is this document about?"
    #     results = loaded_db.similarity_search(query, k=5)
    #     print(results)

    embd_vectordb("ML.pdf")
