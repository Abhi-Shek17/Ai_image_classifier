# AI or Real Image Classifier

### Deployed link: [AI or Real Image Classifier](https://aiimageclassifier-6gxzsacc7er3gmeqky92m7.streamlit.app/)

This repository contains code and resources for an image classifier project. The project is built using a machine learning model to classify images as either AI-generated or real.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Procedure](#procedure)
- [Result](#result)
- [Preview](#preview)

## Project Overview

The image classifier project aims to distinguish between AI-generated images and real images using a pre-trained machine learning model. The project leverages the Streamlit framework to create an interactive web application.

## Installation

To install the necessary dependencies, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the dependencies listed in the `Requirements.txt` file:

   ```sh
   pip install -r Requirements.txt
   ```

## Usage

To run the image classifier application, use the following command:

```sh
streamlit run app.py
```

This will start the Streamlit application, and you can access it in your web browser.

Once the application is running, follow these steps:

1. Click on the "Browse Files" button to select an image file (in JPG, PNG, JPEG, or WEBM format).
2. Wait for the model to load and process the image.
3. The application will display the uploaded image along with the prediction result.
4. The prediction will indicate whether the image is most likely AI-generated or a real image.

## Procedure

### Web Scraping

The dataset (available at [Kaggle](https://www.kaggle.com/datasets/idkwhatodo/classify-ai-or-real)) contains 2376 AI-generated images and 2351 real images, which have been resized to (512, 512) for convenience. These images were web-scraped using search terms such as "men AI-generated images" and "men real images."

### Training

1. **Transfer Learning:** By utilizing transfer learning from the VGG 16 model, the training process has been significantly streamlined.('checkpoint.pth')
2. **Custom Model (myNet):** This model comprises 2 convolutional layers, 2 pooling layers, 2 fully connected layers, and 1 output layer.('mynet.pth')

The training and testing notebooks are available in the second branch of this repository.

## Result

### Testing Results

1. **True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN):** 
   - First method: TP = 205, TN = 243, FP = 72, FN = 26
   - Second method: TP = 191, TN = 252, FP = 17, FN = 86

2. **Accuracy:** 82%

### Comparison
The misclassified set of the first method is more justifiable and comprehensible compared to the second method.

## Preview
### Start 
![ai_1](https://github.com/Abhi-Shek17/Ai_image_classifier/assets/136077817/d008a11a-ba81-4060-88de-7a8a3d460cb5)
### Browse
![ai_2](https://github.com/Abhi-Shek17/Ai_image_classifier/assets/136077817/193c797c-e43f-4891-a6e7-be92040dee85)
### Output for real image
![ai_3](https://github.com/Abhi-Shek17/Ai_image_classifier/assets/136077817/03911825-42c1-47ca-966e-c0bfb2a7df5c)
### Output for AI image
![image](https://github.com/Abhi-Shek17/Ai_image_classifier/assets/136077817/99e1410a-284a-4ab9-a5e5-49afc7e8d6c7)
