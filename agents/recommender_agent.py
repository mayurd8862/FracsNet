from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model="nemotron-mini",
    temperature=0,
    num_predict=256,
)

class RecommenderResponse(BaseModel):
    recommended_questions: List[str] = Field(
        description="List of three follow-up questions related to symptoms, dosage, side effects, and other relevant aspects."
    )

def recommend_query(query: str) -> dict:
    parser = JsonOutputParser(pydantic_object=RecommenderResponse)
    
    prompt = PromptTemplate(
        template="""
        You are a smart AI assistant for a healthcare e-commerce application.
        Your task is to generate three relevant follow-up questions that a user might ask based on their query.
        These questions should focus on symptoms, dosage, side effects, and other important aspects related to the product inquiry.
        Questions should not be lengthy.
        
        {format_instructions}
        Query: {query}
        
        phrase questions naturally from the chatbot's perspective.
        """,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"query": query})
        return result
    except Exception as e:
        print(f"Error in generating recommended questions: {e}")
        return {"recommended_questions": []}


def recommend_product(query: str) -> dict:
    pass
#############################################
#        Data science Team working          #
#############################################


