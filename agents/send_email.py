# import smtplib
# from email.message import EmailMessage

# # SMTP Configuration (Use your credentials)
# SMTP_SERVER = "smtp.gmail.com"  # Gmail SMTP
# SMTP_PORT = 587
# SENDER_EMAIL = "mayurdabade1103@gmail.com"
# SENDER_PASSWORD = "@Mayur2003"  # Use App Password for Gmail

# # Function to Send Email
# def send_order_email(order_details):
#     try:
#         msg = EmailMessage()
#         msg["Subject"] = f"Order Confirmation - {order_details['OrderID']}"
#         msg["From"] = SENDER_EMAIL
#         msg["To"] = order_details["Email"]

#         # Email Body
#         email_body = f"""
#         Hello customer,

#         ✅ Your order for **{order_details["Product"]}** has been placed successfully! 🎉

#         🔹 **Order Details**:
#         - **Order ID:** {order_details["OrderID"]}
#         - **Product:** {order_details["Product"]}
#         - **Quantity:** {order_details["Quantity"]}
#         - **Delivery Address:** {order_details["Address"]}
#         - **Payment Method:** {order_details["PaymentMethod"]}

#         Thank you for shopping with us! 😊
#         """
#         msg.set_content(email_body)

#         # Send Email
#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.starttls()
#             server.login(SENDER_EMAIL, SENDER_PASSWORD)
#             server.send_message(msg)

#         return True  # Email Sent Successfully

#     except Exception as e:
#         print(f"Error sending email: {e}")
#         return False  # Email Sending Failed








import smtplib
from email.message import EmailMessage
import os

# Load credentials securely (Use Environment Variables)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "mayurdabade1103@gmail.com"

from dotenv import load_dotenv
load_dotenv()

SENDER_PASSWORD = os.getenv("EMAIL_PASSWORD")  # Use environment variable for security

# Function to Send Email
def send_order_email(order_details, email):
    try:
        msg = EmailMessage()
        msg["Subject"] = f"Order Confirmation - {order_details['OrderID']}"
        msg["From"] = SENDER_EMAIL
        # msg["To"] = order_details["Email"]
        msg["To"] = email

        # Email Body
        email_body = f"""
        Hello Dr.,

        ✅ Your order for {order_details["Product"]} has been placed successfully! 🎉

        🔹 Order Details:
        - Order ID : {order_details["OrderID"]}
        - Product : {order_details["Product"]}
        - Quantity : {order_details["Quantity"]}
        - Delivery Address : {order_details["Address"]}
        - Payment Method : {order_details["PaymentMethod"]}
        - Timestamp : {order_details["Timestamp"]}

        Thank you for shopping with us! 😊
        """
        msg.set_content(email_body)

        # Send Email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print("✅ Email sent successfully!")
        return True

    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False
    

# # Function to Send Email
# def OTP_verification_email(email,otp):
#     try:
#         msg = EmailMessage()
#         msg["Subject"] = "Your OTP for Registration"
#         msg["From"] = SENDER_EMAIL
#         msg["To"] = email


#         email_body = f"Hello,\n\nYour One-Time Password (OTP) for registration is: {otp}\n\nThis OTP is valid for 10 minutes.\n\nThank you,\nFracsNet Team"
#         msg.set_content(email_body)

#         # Send Email
#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.starttls()
#             server.login(SENDER_EMAIL, SENDER_PASSWORD)
#             server.send_message(msg)

#         print("✅ Email sent successfully!")
#         return True

#     except Exception as e:
#         print(f"❌ Error sending email: {e}")
#         return False


def OTP_verification_email(email, otp):
    try:
        msg = EmailMessage()
        msg["Subject"] = "Your OTP for Registration"
        msg["From"] = SENDER_EMAIL
        msg["To"] = email

        # HTML version
        html_content = f"""
        <html>
            <body>
                <h2>Registration OTP</h2>
                <p>Your One-Time Password is: <strong>{otp}</strong></p>
                <p>Valid for 10 minutes</p>
                <hr>
                <p>FracsNet Team</p>
            </body>
        </html>
        """
        
        # Plain text fallback
        text_content = f"Your OTP is: {otp}\nValid for 10 minutes"

        msg.set_content(text_content)
        msg.add_alternative(html_content, subtype="html")

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print("✅ Email sent successfully!")
        return True

    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")
        return False

# # Example Order Data
if __name__ == "__main__":
#     order_details = {
#         "OrderID": "123456789",
#         "Product": "Herbal Tea",
#         "Quantity": 2,
#         "Address": "123, Green Street, Pune",
#         "Email": "mayur.dabade21@vit.edu",
#         "PaymentMethod": "Credit Card",
#     }
    
#     # Send Order Email
#     # send_order_email(order_details)
    OTP_verification_email("mayur.dabade21@vit.edu", "123456")
