import streamlit as st
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
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
import torch.nn.functional as F
# @st.cache_data
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.ada_pool = nn.AdaptiveAvgPool2d((10, 10))
        self.fc1 = nn.LazyLinear(100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.ada_pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
 
def load_model():
    if 'model' in st.session_state :
        return st.session_state['model']
    model =MyNet()
    model.load_state_dict(torch.load('mystatedic.pth'))
    st.session_state['model']=model
    return model

@st.cache_data
def predict(img):
    model=load_model()
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
    col1,col2 = st.columns(2)
    with col1:
        st.image(img)
    # Display image and caption
    with col2:
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
