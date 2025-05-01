from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

def summary_agent(query: str, llm, vectordb):

    template = f"""Based on the following document excerpts, provide a comprehensive summary about the topic: {{question}} 
    
    DOCUMENT EXCERPTS:
    {{context}} 
    
    Instructions:
    1. Synthesize the main points related to the topic
    2. Organize information in a logical structure
    3. Include key details, facts, and data when relevant
    4. Maintain accuracy without adding information not present in the excerpts
    5. Create a coherent, flowing summary that captures the essential information
    
    SUMMARY:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),  # Limit to top 3 results
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    result = qa_chain.invoke(query)
    return result["result"]


# if __name__ == "__main__":
#     vectordb = load_vectordb()
#     llm = ChatGroq(model_name="Llama3-8b-8192")
#     a = generate_response("who is pm of india", llm, vectordb)
#     print(a)


# uvicorn backend:app --reload --port 5000


