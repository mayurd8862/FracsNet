from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
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

class QueryValidator(BaseModel):
    validated: str = Field(
        description="Query will be validated or not YES or NO"
    )

def validator_agent(query: str) -> dict:
# Initialize the model and parser

    parser = JsonOutputParser(pydantic_object=QueryValidator)

    prompt = PromptTemplate(
        template="""

        You are a helpful AI assistant for a healthcare e-commerce application that provides users with product information, price comparisons, ordering assistance, and personalized recommendations.

        Your task is to determine whether the user’s query is relevant to the healthcare e-commerce platform.

        The user is allowed to:
        1. Ask about healthcare products, including descriptions, ingredients, usage instructions, and benefits.
        2. Request price comparisons between different products or brands.
        3. Make an order for healthcare products.
        4. Ask for personalized recommendations based on their needs.
        5. Inquire about order status, delivery details, and return policies.

        The user is NOT allowed to:
        1. Ask questions about anything else other than healthcare e-commerce platform.
        2. Request personal medical advice or prescriptions.
        3. Ask about topics unrelated to the e-commerce platform.

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

