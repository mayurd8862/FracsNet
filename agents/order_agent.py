import requests
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# llm = ChatOllama(
#     model = "nemotron-mini",
#     temperature = 0,
#     num_predict = 256,
#     # other params ...
# )

llm = ChatGroq(model_name="Llama3-8b-8192")


# Function to call the Groq API
def call_groq_api(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "groq-llm",  # Specify the model used by Groq
        "messages": [{"role": "system", "content": prompt}],
        "max_tokens": 150,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"Error calling Groq API: {response.status_code} - {response.text}")

# Define the dynamic prompt
def create_prompt(product_name, quantity, address):
    return f"""
You are an order-taking assistant for a healthcare e-commerce system. Your task is to engage with the user in a conversation to collect all the required details for an order. You must ensure all the following details are filled in completely before confirming the order:
{{
    "product_name": "{product_name or 'XYZ'}",
    "Quantity": "{quantity or 'N'}",
    "Address": "{address or 'PQR'}"
}}

### Guidelines:
1. Ask for each detail step-by-step, ensuring clarity and avoiding ambiguity.
2. If the user provides incomplete or unclear information, politely ask for clarification.
3. Use a loop: If any required detail is missing, keep asking about it until all fields are filled.
4. Confirm each detail with the user after collecting it to avoid errors.
5. Use a friendly, helpful tone to maintain a positive user experience.

### Output Format:
Collect the final details in this structured format:
{{
    "product_name": "{product_name}",
    "Quantity": "{quantity}",
    "Address": "{address}"
}}

### Current Status:
- Product Name: {"Provided" if product_name else "Missing"}
- Quantity: {"Provided" if quantity else "Missing"}
- Address: {"Provided" if address else "Missing"}

Based on the above, ask for the missing details.
"""

# Order-taking agent logic
def order_taking_agent():
    product_name = None
    quantity = None
    address = None

    while not (product_name and quantity and address):
        # Create the dynamic prompt based on missing details
        prompt = create_prompt(product_name, quantity, address)

        # Call the Groq API
        try:
            message = call_groq_api(prompt)
            print(message)
        except Exception as e:
            print(f"Error: {e}")
            break

        # Simulate user input based on the response
        if "product" in message.lower():
            product_name = input("User: ")
        elif "quantity" in message.lower():
            quantity = input("User: ")
        elif "address" in message.lower():
            address = input("User: ")

    # Confirm the order details with the user
    print(f"\nGreat! Here's what I have for your order:")
    print(f"Product Name: {product_name}")
    print(f"Quantity: {quantity}")
    print(f"Address: {address}")
    confirm = input("\nIs everything correct? (yes/no): ")

    if confirm.lower() != "yes":
        print("Let's correct the details. Restarting the process...\n")
        order_taking_agent()
    else:
        print("Thank you for your order! It has been placed successfully.")

# Run the order-taking agent
if __name__ == "__main__":
    order_taking_agent()
