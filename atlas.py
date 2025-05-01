import pymongo
import base64

client = pymongo.MongoClient("mongodb+srv://mayurr:12345@cluster0.hllwy4r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client["fracsnet"]
products_collection = db["products"]


def get_product_names():
    """Fetch all product names from MongoDB and cache them."""
    return [doc["product_name"] for doc in products_collection.find({}, {"product_name": 1})]

def get_product_image(product_name):
    """Fetch the base64-encoded image from MongoDB and decode it."""
    product_data = products_collection.find_one({"product_name": product_name}, {"image_base64": 1})
    
    if product_data and "image_base64" in product_data:
        return base64.b64decode(product_data["image_base64"])
    return None

from PIL import Image
from io import BytesIO
import base64

def show_product_image(product_name):
    """Fetch and display the product image from MongoDB."""
    image_data = get_product_image(product_name)
    
    if image_data:
        img = Image.open(BytesIO(image_data))
        img.show()  # Opens the image in the default viewer
    else:
        print("No image found for the given product.")


def save_product_image(product_name, filename="output_image.jpg"):
    """Fetch and save the product image from MongoDB."""
    image_data = get_product_image(product_name)
    
    if image_data:
        with open(filename, "wb") as img_file:
            img_file.write(image_data)
        print(f"Image saved as {filename}")
    else:
        print("No image found for the given product.")


# Example usage:
show_product_image("Viracid")


# print(get_product_image("Viracid"))