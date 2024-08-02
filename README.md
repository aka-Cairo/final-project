# Chest X-Ray Multi-Label Classification

This repository contains a project for classifying chest X-ray images using a ResNet-50 model. The goal is to predict multiple pathologies from grayscale chest X-ray images. The project is divided into two main parts: training the model and deploying it with a Gradio app for interactive predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Training the Model](#training-the-model)
- [Gradio App](#gradio-app)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project involves training a ResNet-50 model from scratch to classify multiple pathologies in chest X-ray images. The trained model is then used in a Gradio app to provide real-time predictions and descriptions of detected pathologies.

## Dataset

- **Grayscale Chest X-Ray Images:** Approximately 5000 grayscale images of size 1024x1024.
- **Labels:** Multi-label format indicating the presence of 14 different pathologies.

## Model

The model used is a ResNet-50 architecture, which has been modified to handle grayscale images and output predictions for 14 pathologies.

### Model Details

- **Architecture:** ResNet-50
- **Input:** Grayscale images of size 224x224
- **Output:** 14-pathology classification

## Training the Model

The model is trained using the `resnet50_scratch.ipynb` Jupyter Notebook. The training process includes:

1. **Data Preparation:** Loading and preprocessing the chest X-ray images and their corresponding labels.
2. **Model Definition:** Creating a modified ResNet-50 model to handle grayscale images.
3. **Training:** Optimizing the model using a loss function suitable for multi-label classification and evaluating its performance.

To train the model, execute the `resnet50_scratch.ipynb` notebook. Ensure that all necessary libraries are installed.

## Gradio App
The Gradio app is implemented in the gradio_openai.ipynb Jupyter Notebook. It allows users to upload chest X-ray images and get predictions from the trained model.

### Features

 - **Upload Images:** Users can upload chest X-ray Images for prediction.
 - **Display Predictions:** The app outputs a vector of predictions and a description of detected pathologies using OpenAI's GPT.

 To run the Gradio app, execute the gradio_openai.ipynb notebook. The app will prompt for image uploads and display results interactively.


## Usage
1. Train the Model:
Open and execute the resnet50_scratch.ipynb notebook to train the model.
2. Run the Gradio App:
Open and execute the gradio_openai.ipynb notebook.
Interact with the Gradio app to upload images and receive predictions.

## Acknowledgements
- **ResNet Architecture:** Originally proposed by Kaiming He et al.

- **Gradio:** For creating an interactive web interface.

- **OpenAI GPT:** For generating descriptive text about detected pathologies.