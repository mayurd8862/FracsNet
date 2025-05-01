from pydantic import BaseModel, EmailStr
from typing import List

class UserRegister(BaseModel):
    name: str
    username: str
    password: str
    mail: EmailStr
    usertype: str
    region: str

class UserLogin(BaseModel):
    username: str
    password: str

class Order(BaseModel):
    product: str
    quantity: int
    address: str
    payment_method: str
