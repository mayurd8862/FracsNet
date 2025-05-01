from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from functools import lru_cache
from agents.router_agent import router_agent
# Load environment variables
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from fastapi.middleware.cors import CORSMiddleware  



app = FastAPI(
    title="QA System API",
    description="API for question answering system using LangChain and vector database",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (use specific origins in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Cached LLM initialization
@lru_cache()
def get_llm():
    return ChatGroq(model_name="Llama3-8b-8192")

# Cached vector database initialization
@lru_cache()
def get_vectordb():
    embedding = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Explicitly set to CPU if you don't need GPU
    )
    return Chroma(persist_directory="./test_db", embedding_function=embedding)

class Query(BaseModel):
    question: str

def generate_response(query: str):
    template = f"""Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use 3 sentences maximum. Keep the answer as concise as possible. 

    {{context}} 
    Question: {{question}} 
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    try:
        llm = get_llm()
        vectordb = get_vectordb()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),  # Limit to top 3 results
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        result = qa_chain.invoke(query)
        return result["result"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the QA System API"}

@app.post("/ask")
async def ask_question(query: Query):
    try:
        routed_agent = router_agent(query)["intent"]
        if routed_agent == "INFO":
            answer = generate_response(
                query.question
            )
            return {
                "question": query.question,
                "answer": answer
            }
        else:
            ans = f"Query will be routed to the {routed_agent} agent"
            return {
                "question": query.question,
                "answer": ans
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, workers=1)


# uvicorn backend:app --reload --port 5000





