from langchain_groq import ChatGroq
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model = "nemotron-mini",
    temperature = 0,
    num_predict = 256,
    # other params ...
)

# llm = ChatGroq(model_name="Llama3-8b-8192")

class QueryClassification(BaseModel):
    intent: str = Field(
        description="Primary intent category (ORDER, COMPARE, RECOMMEND, INFO)"
    )

def router_agent(query: str) -> dict:
# Initialize the model and parser

    parser = JsonOutputParser(pydantic_object=QueryClassification)

    prompt = PromptTemplate(
        template="""

        You are a helpful AI assistant for a healthcare e-commerce application.
        Your task is to determine which agent should handle the user input. You have 4 agents to choose from:
        1. ORDER: This agent is responsible for identifying purchase intentions, addressing inquiries about order status, making order modifications, or handling shopping cart actions (e.g., view, add, remove, modify items).
        2. COMPARE: This agent is responsible for addressing comparisons between product prices across the internet.
        3. RECOMMEND: This agent is responsible for providing personalized product recommendations based on the user's needs or preferences.
        4. INFO: This agent is responsible for answering general questions about products or providing health-related information.

        {format_instructions}
        Query: {query}

        Your output should be in a structured JSON format like so. Each key is a string and each value is a string.

        """,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )



    # Create the classification chain
    chain = prompt | llm | parser

    try:
        result = chain.invoke({"query": query})
        return result
    except Exception as e:
        print(f"Error in character information extraction: {e}")
        return {}

