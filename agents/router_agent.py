# from typing import List
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama
# from dotenv import load_dotenv

# load_dotenv()

# llm = ChatOllama(
#     model = "nemotron-mini",
#     temperature = 0,
#     num_predict = 256,
#     # other params ...
# )

# # llm = ChatGroq(model_name="Llama3-8b-8192")

# class QueryClassification(BaseModel):
#     intent: str = Field(
#         description="Primary intent category (ORDER, COMPARE, RECOMMEND, INFO, SUMMARY)"
#     )

# def router_agent(query: str) -> dict:
# # Initialize the model and parser

#     parser = JsonOutputParser(pydantic_object=QueryClassification)

#     prompt = PromptTemplate(
#         template="""

#         You are a helpful AI assistant for a healthcare e-commerce application.
#         Your task is to determine which agent should handle the user input. You have 4 agents to choose from:
#         1. ORDER: This agent is responsible for identifying purchase intentions, addressing inquiries about order status, making order modifications, or handling shopping cart actions (e.g., view, add, remove, modify items).
#         2. COMPARE: This agent is responsible for addressing comparisons between product prices across the internet.
#         3. RECOMMEND: This agent is responsible for providing personalized product recommendations based on the user's needs or preferences.
#         4. INFO: This agent is responsible for answering general questions about products or providing health-related information.
#         5. SUMMARY: Identifies when a user requests summarization of **product details, health articles, or any lengthy information**.  
#         - If a user asks for **summarization** (e.g., "Summarize this", "Give me a shorter version", "Make this concise"), classify it as `"SUMMARY"`.
#         - **DO NOT** classify general questions as `"SUMMARY"` unless they explicitly mention summarization.

#         {format_instructions}
#         Query: {query}

#         Your output should be in a structured JSON format like so. Each key is a string and each value is a string.

#         """,

#         input_variables=["query"],
#         partial_variables={"format_instructions": parser.get_format_instructions()}
#     )

#     # Create the classification chain
#     chain = prompt | llm | parser

#     try:
#         result = chain.invoke({"query": query})
#         return result
#     except Exception as e:
#         print(f"Error in character information extraction: {e}")
#         return {}

# # print(router_agent("I want to order product"))








from typing import Literal
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
# llm = ChatOllama(
#     model="nemotron-mini",
#     temperature=0.2,
#     num_predict=256
# )

# If needed, swap to Groq:
from langchain_groq import ChatGroq
llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct")

# Define allowed intent categories
class QueryClassification(BaseModel):
    intent: Literal["ORDER", "COMPARE", "RECOMMEND", "INFO", "SUMMARY", "NONE"] = Field(
        description="Primary intent category (ORDER, COMPARE, RECOMMEND, INFO, SUMMARY, or NONE for greetings/small talk)"
    )

def router_agent(query: str) -> dict:
    # Parser setup
    parser = JsonOutputParser(pydantic_object=QueryClassification)

    # Prompt for classifying the query
    classification_prompt = PromptTemplate(
        template="""
You are a helpful AI assistant for a healthcare e-commerce application.
Classify the user's intent into one of the following categories:

1. ORDER: For purchase actions, cart, or order status.
2. COMPARE: Comparing product prices.
3. RECOMMEND: Recommending products.
4. INFO: General product or health-related info.
5. SUMMARY: If the user explicitly asks to summarize something.
6. NONE: For greetings or casual talk like "hi", "how are you", "who are you".

{format_instructions}
Query: {query}

Respond with a JSON object containing only the `"intent"` key and its value.
""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Build chain for classification
    classify_chain = classification_prompt | llm | parser

    try:
        classification = classify_chain.invoke({"query": query})
        intent = classification["intent"]

        # If intent is NONE, generate a direct LLM response
        if intent == "NONE":
            casual_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are Emma, a friendly and intelligent AI assistant for a healthcare e-commerce platform. You specialize in providing helpful and accurate responses to users’ healthcare-related queries, including product recommendations, order tracking, pricing, and basic medical information (not diagnosis). You respond in a warm, approachable tone and are always polite and supportive.

                In casual interactions such as greetings or small talk, you behave like a friendly companion—engaging naturally while maintaining professionalism. When the user asks a question or makes a request, you shift into a helpful and informative tone, offering clear and concise responses. You are capable of understanding context and guiding users to make informed choices about healthcare products or services.
                 
                Answer should be short and concise, but also friendly and engaging. You can use emojis to make the conversation more lively and relatable.

                Your primary goals are to:
                Be approachable and friendly during casual conversations.
                Be knowledgeable and efficient during queries related to healthcare products, orders, or platform usage.
                Encourage trust and comfort in users while handling sensitive topics."""),
                ("human", "{query}")
            ])
            chat_chain = casual_prompt | llm
            response = chat_chain.invoke({"query": query})
            return {
                "intent": "NONE",
                "response": response.content
            }

        # For all other intents, return the intent only
        return {
            "intent": intent
        }

    except Exception as e:
        print(f"Error in router agent: {e}")
        return {"intent": "NONE", "response": "Oops! I had trouble understanding that. Could you try again?"}
