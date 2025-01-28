from langchain_groq import ChatGroq
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "llama3.2",
    temperature = 0,
    num_predict = 256,
    # other params ...
)

# Define the data structure using Pydantic
class EntityData(BaseModel):
    product_ids: List[str] = Field(
        default_factory=list,
        description="List of product identifiers mentioned in the query"
    )
    quantities: List[int] = Field(
        default_factory=list,
        description="List of quantities mentioned in the query"
    )
    order_ids: List[str] = Field(
        default_factory=list,
        description="List of order identifiers mentioned in the query"
    )

class QueryClassification(BaseModel):
    primary_intent: str = Field(
        description="Primary intent category (ORDER, CART, COMPARE, RECOMMEND, INFO)"
    )
    secondary_intents: List[str] = Field(
        default_factory=list,
        description="List of secondary intents identified in the query"
    )
    confidence: float = Field(
        description="Confidence score for the classification (0.0 to 1.0)"
    )
    entities: EntityData = Field(
        description="Structured data extracted from the query"
    )
    requires_context: bool = Field(
        description="Whether additional context is needed to fully process the query"
    )
    

def route_info(query: str) -> dict:
# Initialize the model and parser

    parser = JsonOutputParser(pydantic_object=QueryClassification)

    prompt = PromptTemplate(
        template="""
        You are an expert input classifier for a healthcare e-commerce system. Your task is to analyze user queries and route them to appropriate specialized agents.

        Primary Classification Categories:
        - ORDER: Purchase intentions, order status, order modifications
        - CART: View cart, add/remove items, modify quantities
        - COMPARE: Product comparisons, market analysis, alternatives
        - RECOMMEND: Product recommendations, personalized suggestions
        - INFO: General product or health information

        Classification Rules:
        1. ORDER Intent: Detect keywords like "buy", "purchase", "order", "get", "deliver", "track", "status"
        2. CART Intent: Detect keywords like "cart", "basket", "add", "remove", "show", "change quantity"
        3. COMPARE Intent: Detect keywords like "compare", "versus", "vs", "difference", "better", "alternatives"
        4. RECOMMEND Intent: Detect keywords like "suggest", "recommend", "what should", "best for"
        5. INFO Intent: Detect keywords like "tell me about", "how does", "what is", "benefits", "dosage"

        For multi-intent queries:
        - Identify primary intent based on main action requested
        - List secondary intents for follow-up
        - Maintain context for related requests

        {format_instructions}

        Query: {query}
        """,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # llm = ChatGroq(model_name="Llama3-8b-8192")

    # Create the classification chain
    chain = prompt | llm | parser

    try:
        result = chain.invoke({"query": query})
        return result
    except Exception as e:
        print(f"Error in character information extraction: {e}")
        return {}











