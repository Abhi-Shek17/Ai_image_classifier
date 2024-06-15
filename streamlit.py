import streamlit as st
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import base64
import io

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

@st.cache_data
def predict(img):
    model = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ])
    pil_image = Image.open(img).convert("RGB")
    img_tensor = transform(pil_image)
    return model(img_tensor.unsqueeze(dim=0))

img = st.file_uploader("Input your image", type=['jpg', 'png'])
class_name = ['AI generated image', 'Real image']

if img is not None:
    idx = torch.argmax(predict(img))
    string = class_name[idx]

    # Convert the uploaded image to base64
    pil_image = Image.open(img).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    # Display image and caption
    st.markdown(
        f"""
        <div class="image-container">
            <img src="data:image/png;base64,{img_str}" alt="Uploaded Image">
            <div class="caption-container">
                <h2 class="caption" >{string}</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )