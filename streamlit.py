import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import requests
import os

# Define custom CSS for the background, title, and caption
st.markdown(
    """
    <style>
    body {
        background-color: transparent;
        background-image: url('https://www.futuroprossimo.it/wp-content/uploads/2023/09/1696077112517.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .title {
        font-size: 68px;
        font-weight: bold;
        text-align: center;
        color: #000000;
        margin-top: 20px;
    }
    .caption {
        font-size: 24px;
        text-align: center;
        color: #FF5733;
        margin-top: 20px;
    }
    .stApp {
        background-color: rgba(211, 211, 211, 0.8); /* Light grey with some transparency */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin: 20px;
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    .image-container img {
        max-width: 300px;
        margin-right: 20px;
    }
    .caption-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .caption {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #000000;
        margin-top: 20px;
    }
    </style>
    <div class="title">AI Image Classifier</div>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def download_model(url, model_path):
    # Download the model file
    # Download the model file
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    with open(model_path, 'wb') as f:
        f.write(response.content)
    # Load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode
    return model

@st.cache_data
def predict(img, model):
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ])
    # Open the image and apply the transformations
    pil_image = Image.open(img).convert("RGB")
    img_tensor = transform(pil_image)
    # Perform the prediction
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(dim=0))
    return output

# Define the URL to the model file on GitHub
model_url = 'https://github.com/Abhi-Shek17/Ai_image_classifier/tree/main/checkpoint.py'
model_path = 'model.pth'

# Download and load the model
model = download_model(model_url, model_path)

# File uploader to allow users to upload images
img = st.file_uploader("Input your image", type=['jpg', 'png'])
class_name = ['AI generated image', 'Real image']

if img is not None:
    # Make the prediction
    idx = torch.argmax(predict(img, model)).item()
    string = class_name[idx]
    # Display the uploaded image
    st.image(img)
    # Display the prediction result as a caption
    st.markdown(
        f"""
        <div class="image-container">
            <div class="caption-container">
                <h2 class="caption" >{string}</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
