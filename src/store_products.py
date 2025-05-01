import pandas as pd
import base64
import pymongo
import requests
from io import BytesIO
from PIL import Image
import numpy as np

# MongoDB connection setup
# client = pymongo.MongoClient("mongodb://localhost:27017/")
client = pymongo.MongoClient("mongodb+srv://mayurr:12345@cluster0.hllwy4r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["fracsnet"]
collection = db["products"]

# Load CSV file
csv_file = "C:/Users/mayur/Desktop/FRACSNET/knowledge/supplements_data.csv"  # Update to your CSV path
df = pd.read_csv(csv_file)

# Replace NaN values with an empty string
df.fillna("", inplace=True)

# Convert local image to base64
def local_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error loading local image {image_path}: {e}")
        return None

# Convert Google Drive URL to direct download URL
def convert_drive_url(gdrive_url):
    if "drive.google.com" in gdrive_url:
        try:
            file_id = gdrive_url.split("id=")[1]  # Extract file ID
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        except IndexError:
            print(f"Invalid Google Drive URL: {gdrive_url}")
            return None
    return gdrive_url  # Return original URL if not Google Drive

# Convert image URL to base64
def url_to_base64(image_url):
    try:
        # Convert Google Drive URL if needed
        image_url = convert_drive_url(image_url)

        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            buffered = BytesIO()
            img.save(buffered, format="PNG")  # Ensure it's saved in a proper format
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            print(f"Failed to download image: {image_url}, Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error processing image from URL {image_url}: {e}")
        return None

# Process each row in the DataFrame
for index, row in df.iterrows():
    product_name = str(row["ProductName"])  # Ensure it's a string
    image_path = str(row["Product_Image"])  # Convert NaN to empty string

    if image_path.strip():  # Skip empty values
        # Check if the image is a URL
        if image_path.startswith("http"):
            base64_image = url_to_base64(image_path)
        else:
            base64_image = local_image_to_base64(image_path)

        if base64_image:
            # Prepare document for MongoDB
            product_data = {
                "product_name": product_name,
                "image_base64": base64_image
            }

            # Insert into MongoDB
            collection.insert_one(product_data)
            print(f"Inserted: {product_name}")
    else:
        print(f"Skipping: {product_name} (No Image Provided)")

# print("Data successfully stored in MongoDB!")


