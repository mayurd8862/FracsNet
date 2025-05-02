# ######################################################
# #    Responce Generation Using ChromaDB              #
# ######################################################

# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain_chroma import Chroma
# from functools import lru_cache

# # Cached LLM initialization
# @lru_cache()
# def get_llm():
#     return ChatGroq(model_name="Llama3-8b-8192")

# # Cached vector database initialization
# @lru_cache()
# def get_vectordb():
#     embedding = SentenceTransformerEmbeddings(
#         model_name="all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'}  # Explicitly set to CPU if you don't need GPU
#     )
#     return Chroma(persist_directory="./test_db", embedding_function=embedding)

# def generate_response(query: str):
#     """
#     Generates a response to the user's query using the info agent.
#     """
#     template ="""You are a healthcare e-commerce assistant that provides factual, direct answers based solely on the provided context. 

#     IMPORTANT: Do not add greetings, introductions, or closing questions when responding to direct queries. Only respond with relevant information from the context.

#     RULES:
#     - If the user's message is a greeting (like "hi", "hello", "hey","how are u" etc.) or contains only small talk, respond with a friendly greeting
#     - Answer directly without adding "Hi there" or "I'm happy to help" introductions
#     - Do not ask follow-up questions like "Do you have any other questions?"
#     - Only acknowledge greetings if the user's message is purely a greeting with no question
#     - Use simple, patient-friendly language while being factual
#     - Only use information found in the context
#     - Say "I don't have enough information to answer that" if the context doesn't contain relevant information

#     Context:
#     {context}
    
#     Patient's Question:
#     {question}
#     """

#     QA_CHAIN_PROMPT = PromptTemplate(
#         input_variables=["context", "question"],
#         template=template
#     )

#     try:
#         llm = get_llm()
#         vectordb = get_vectordb()
        
#         qa_chain = RetrievalQA.from_chain_type(
#             llm,
#             retriever=vectordb.as_retriever(search_kwargs={"k": 3}),  # Limit to top 3 results
#             return_source_documents=True,
#             chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#         )

#         result = qa_chain.invoke(query)
#         return result["result"]
#     except Exception as e:
#         raise Exception(f"Error generating response: {str(e)}")




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


# def response_generatorr(vectordb, query, llm):
#     # vectordb = get_vectordb()
#     # llm = get_llm()
#     context = vectordb.similarity_search(query, k=4)

#     # for res in results:
#     #     print(f"📄 {res.page_content} [{res.metadata}]")
#     template = f"""
#     You are an intelligent assistant designed to provide accurate and concise answers based on the context provided. 
#     Follow these rules strictly:
#     1. Use ONLY the information provided in the context to answer the question.
#     2. If the context does not contain enough information to answer the question, say "I don't know."
#     3. Do not make up or assume any information outside of the context.
#     4. Keep your answer concise and to the point (maximum 3 sentences).

#     Context:
#     {context}

#     Question:
#     {query}

#     Helpful Answer:
#     """
#     ans = llm.invoke(template)
#     return ans.content


def response_generator(query):
        template ="""You are a healthcare e-commerce assistant that provides factual, direct answers based solely on the provided context. 

        IMPORTANT: Do not add greetings, introductions, or closing questions when responding to direct queries. Only respond with relevant information from the context.

        RULES:
        - If the user's message is a greeting (like "hi", "hello", "hey","how are u" etc.) or contains only small talk, respond with a friendly greeting
        - Answer directly without adding "Hi there" or "I'm happy to help" introductions
        - Do not ask follow-up questions like "Do you have any other questions?"
        - Only acknowledge greetings if the user's message is purely a greeting with no question
        - Use simple, patient-friendly language while being factual
        - Only use information found in the context
        - Say "I don't have enough information to answer that" if the context doesn't contain relevant information

        Context:
        {context}
        
        Patient's Question:
        {question}
        """

        llm = get_llm()
        vectordb = get_vectordb()

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        qa_chain = RetrievalQA.from_chain_type(llm, 
                                            retriever=vectordb.as_retriever(), 
                                            return_source_documents=True, 
                                            chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

        ans = qa_chain.invoke({"query": query})
        return ans["result"]
