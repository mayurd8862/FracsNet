import smtplib
from email.message import EmailMessage
import streamlit as st
import math, random

# function to generate OTP
def generateOTP() :
    digits = "0123456789"
    OTP = ""

    for i in range(4) :
        OTP += digits[math.floor(random.random() * 10)]
 
    return OTP