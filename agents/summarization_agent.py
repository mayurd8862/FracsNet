# # ######################################################
# # #    Responce Generation Using ChromaDB              #
# # ######################################################


# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain_chroma import Chroma
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from dotenv import load_dotenv
# load_dotenv()

# def summary_agent(query: str, llm, vectordb):

#     template = f"""Based on the following document excerpts, provide a comprehensive summary about the topic: {{question}} 
    
#     DOCUMENT EXCERPTS:
#     {{context}} 
    
#     Instructions:
#     1. Synthesize the main points related to the topic
#     2. Organize information in a logical structure
#     3. Include key details, facts, and data when relevant
#     4. Maintain accuracy without adding information not present in the excerpts
#     5. Create a coherent, flowing summary that captures the essential information
    
#     SUMMARY:"""

#     QA_CHAIN_PROMPT = PromptTemplate(
#         input_variables=["context", "question"],
#         template=template
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm,
#         retriever=vectordb.as_retriever(search_kwargs={"k": 3}),  # Limit to top 3 results
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#     )

#     result = qa_chain.invoke(query)
#     return result["result"]



# ######################################################
# #          Responce Generation Using Qdrant          #
# ######################################################





from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
import qdrant_client
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from uuid import uuid4
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.runnables import Runnable

from dotenv import load_dotenv
load_dotenv()
qdrant_api_key = os.getenv("QDRANT_API_KEY")

from functools import lru_cache


# Cached LLM initialization
@lru_cache()
def get_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile")

@lru_cache()
def get_vectordb():

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    client = QdrantClient(path="qdrant_data")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="demo_collection",
        embedding=embeddings,
    )
    return vector_store


def summary_agent(query):

        template = """Based on the following document excerpts, provide a comprehensive summary about the topic: {question} 
    
        DOCUMENT EXCERPTS:
        {context}
        
        Instructions:
        1. Synthesize the main points related to the topic
        2. Organize information in a logical structure
        3. Include key details, facts, and data when relevant
        4. Maintain accuracy without adding information not present in the excerpts
        5. Create a coherent, flowing summary that captures the essential information
        
        SUMMARY:"""

        llm = get_llm()
        vectordb = get_vectordb()

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        qa_chain = RetrievalQA.from_chain_type(llm, 
                                            retriever=vectordb.as_retriever(), 
                                            return_source_documents=True, 
                                            chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

        ans = qa_chain.invoke({"query": query})
        return ans["result"]
