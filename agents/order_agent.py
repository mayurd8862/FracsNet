# import time
# import os
# import requests
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# class OrderTakingAssistant:
#     def __init__(self, model_name="Llama3-8b-8192"):
#         """Initialize the order-taking assistant."""
#         self.llm = ChatGroq(model_name=model_name)
#         self.order_details = {
#             "product_name": None,
#             "Quantity_of_product": None,
#             "Address": None,
#         }
      
#     def is_order_complete(self):
#         """Check if all required order details are provided."""
#         return all(value is not None for value in self.order_details.values())

#     def get_missing_details_prompt(self):
#         """Generate a prompt listing missing order details."""
#         missing_fields = [key for key, value in self.order_details.items() if value is None]
#         return f"The following details are missing: {', '.join(missing_fields)}. Please provide them one by one."
    
#     def get_next_prompt(self):
#         """Generate the next prompt based on missing details."""
#         # Get current state of the order
#         missing_fields = [key for key, value in self.order_details.items() if value is None]
#         first_missing = missing_fields[0] if missing_fields else None
        
#         # Build context from existing details
#         filled_details = {k: v for k, v in self.order_details.items() if v is not None}
#         context = ""
#         if filled_details:
#             context = "So far you've provided:\n" + "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in filled_details.items()])
        
#         # Field-specific instructions
#         field_instructions = {
#             "product_name": "Please provide the exact name of the healthcare product you want to order. Be specific about brand, model, and specifications if applicable.",
#             "Quantity_of_product": "Please specify the quantity needed as a number. If you need multiple sizes or variations, please indicate the quantity for each.",
#             "Address": "Please provide your complete delivery address including street address, city, state/province, postal code, and country."
#         }
        
#         specific_instruction = field_instructions.get(first_missing, "") if first_missing else ""
         
#         prompt = f"""
#         You are an expert order-taking assistant for a healthcare e-commerce system with deep knowledge of medical products.
        
#         CONTEXT:
#         - You are speaking directly to a healthcare customer
#         - Your sole purpose is to collect complete and accurate order information
#         - The user is trying to place an order, but you need {len(missing_fields)} more details
#         - {context}
        
#         MISSING INFORMATION:
#         {self.get_missing_details_prompt()}
        
#         NEXT REQUIREMENT:
#         {specific_instruction}
        
#         GUIDELINES:
#         - Focus exclusively on collecting the NEXT missing field: {first_missing if first_missing else 'None'}
#         - Ask ONE clear, specific question to obtain the exact information needed
#         - If the user provides ambiguous or incomplete information, ask for clarification
#         - Validate information when possible (check if quantities are reasonable numbers, if addresses have all components)
#         - Acknowledge previous information provided before asking for new information
#         - Be concise but helpful - avoid unnecessary text
#         - DO NOT suggest specific products or make assumptions about what the user wants
#         - DO NOT generate fictional order information
#         - DO NOT ask for multiple pieces of information in a single response
        
#         RESPONSE FORMAT:
#         - Brief acknowledgment of any information just provided (1 sentence maximum)
#         - A single clear question focusing only on the next missing detail
#         - Optional brief clarification of what format the information should be in (when applicable)
        
#         EXAMPLE QUALITY RESPONSES:
#         - "Thank you. What specific healthcare product would you like to order today? Please include brand and model if applicable."
#         - "Got it. How many units of this item do you need? Please specify the quantity as a number."
#         - "Thank you. Please provide your complete delivery address including street, city, state, and postal code."
#         """
        
#         # Generate response using ChatGroq (Llama3-8B)
#         try:
#             response = self.llm.invoke(prompt)
#             return response.content.strip()
#         except Exception as e:
#             return f"Error: {str(e)}"
    
#     def process_input(self, user_input):
#         """Process user input and update order details."""
#         for key in self.order_details.keys():
#             if self.order_details[key] is None and user_input:
#                 self.order_details[key] = user_input
#                 break  # Update one field at a time
        
#         # Return the next prompt if order is not complete
#         if not self.is_order_complete():
#             return self.get_next_prompt()
#         else:
#             return "Order complete! Thank you for providing all the details."
    
#     def get_order_details(self):
#         """Return the current order details."""
#         return self.order_details
    
#     def reset(self):
#         """Reset the order details."""
#         self.order_details = {
#             "product_name": None,
#             "Quantity_of_product": None,
#             "Address": None,
#         }

# # # Example usage
# # if __name__ == "__main__":
# #     assistant = OrderTakingAssistant()
    
# #     while not assistant.is_order_complete():
# #         # Get the next prompt from the assistant
# #         prompt = assistant.get_next_prompt()
# #         print(f"Agent: {prompt}")
        
# #         # Simulate user input (replace with actual chatbot input in production)
# #         user_input = input("User: ")
        
# #         # Process the user input
# #         response = assistant.process_input(user_input)
# #         print(f"Agent: {response}")
    
# #     # Display final order details
# #     print("\n✅ Order Complete! Final Details:")
# #     print(assistant.get_order_details())





import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pydantic import BaseModel
from typing import List

class OrderDetails(BaseModel):
    product: str
    quantity: int
    address: str
    email: str

def send_order_email(order: OrderDetails):
    sender_email = "your-email@gmail.com"  # Replace with your email
    sender_password = "your-password"      # Replace with your app password

    subject = "Order Confirmation"
    body = f"""
    Hello,

    Your order has been placed successfully.

    Order Details:
    - Product: {order.product}
    - Quantity: {order.quantity}
    - Delivery Address: {order.address}

    Thank you for shopping with us!

    Regards,
    Order Assistant
    """

    # Setup email message
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = order.email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        # Connect to Gmail SMTP server and send email
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, order.email, msg.as_string())
        server.quit()
        return {"message": "Order confirmation email sent successfully"}
    except Exception as e:
        return {"error": f"Failed to send email: {e}"}

