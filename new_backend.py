from fastapi import FastAPI
import pandas as pd
from mongodb import register_user, login_user
from src.database import get_user_details, get_user_orders, register_user,login_user, place_order, get_product_names, get_product_image
from agents.router_agent import router_agent
from agents.info_agent import generate_response
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from agents.send_email import send_order_email, OTP_verification_email
from typing import Optional
import random

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (replace with your frontend URL in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


class UserLogin(BaseModel):
    username: str
    password: str

class UserOrders(BaseModel):  # Fixed class name to follow naming conventions
    username: str

class UserRegister(BaseModel):
    name: str
    username: str
    password: str
    mail: str
    usertype: str
    region: str

class OrderCreate(BaseModel):
    email: EmailStr
    product: str
    quantity: int
    address: str
    paymentmethod: str

class SendOrderMail(BaseModel):
    email: EmailStr
    otp: int

class Query(BaseModel):
    question: str

# Feedback model
class Feedback(BaseModel):
    query: str
    response: str
    feedback: bool
    suggestion: Optional[str] = Field(None, description="Required if feedback is False")

    @classmethod
    def validate_suggestion(cls, values):
        if values["feedback"] is False and not values["suggestion"]:
            raise ValueError("Suggestion is required when feedback is False.")
        return values

# def load_data():
#     return pd.read_csv("./knowledge/supplements_data.csv")

# df = load_data()

def generateOTP():
    return str(random.randint(100000, 999999))


@app.get("/")
async def root():
    return {"message": "Welcome to the QA System API"}

@app.get("/generate-otp")
def get_otp():
    return {"otp": generateOTP()}

@app.get("/products")
def get_products():
    return get_product_names()


@app.get("/users/{username}/orders")
async def retrieve_user_orders(username: str):
    result = get_user_orders(username)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


@app.get("/user_details/{username}")
async def get_user(username: str):
    result = get_user_details(username)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result

@app.get("/get_image/{product_name}")
async def get_image(product_name: str):
    """Fetches the product image from MongoDB and returns it as a response."""
    image_data = get_product_image(product_name)
    
    if image_data:
        return Response(content=image_data, media_type="image/png")  # Change to "image/jpeg" if needed
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@app.post("/register")
async def create_user(user_data: UserRegister):
    result = register_user(
        name=user_data.name,
        username=user_data.username,
        password=user_data.password,
        mail=user_data.mail,
        usertype=user_data.usertype,
        region=user_data.region
    )
    
    # Check if registration was successful
    if isinstance(result, str):
        # If result is a string, it's an error message
        raise HTTPException(status_code=400, detail=result)
    elif "error" in result:
        # If result has an error key
        raise HTTPException(status_code=400, detail=result["error"])
    
    # Return success response
    return result

@app.post("/login/")
def login(user: UserLogin):
    result = login_user(user.username, user.password)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    else:
        return result
    
@app.post("/send_otp_mail/")
def send_otp_mail(user: SendOrderMail):
    result = OTP_verification_email(user.email,user.otp)
    
    if result == True:
        return result
    else:
        raise HTTPException(status_code=400, detail="mail error")


@app.post("/users/{username}/orders")
async def create_order(username: str, order: OrderCreate):
    result = place_order(
        username=username,
        email=order.email,
        product=order.product,
        quantity=order.quantity,
        address=order.address,
        paymentmethod=order.paymentmethod
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.post("/ask")
async def ask_question(query: Query):
    try:
        routed_agent = router_agent(query.question)["intent"]
        if routed_agent == "INFO":
            answer = generate_response(query.question)
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


@app.post("/feedback")
async def submit_feedback(feedback: Feedback):
    """
    API endpoint to collect user feedback on system responses.
    """
    # Validate suggestion if feedback is False
    if not feedback.feedback and not feedback.suggestion:
        raise HTTPException(status_code=400, detail="Suggestion is required when feedback is False.")

    # Process feedback (store it in a database or log file)
    return {"message": "Feedback submitted successfully", "data": feedback.dict()}
