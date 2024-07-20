# Bone Fracture Image Classification using CNN and Streamlit

This repository contains a project for classifying bone fracture images using Convolutional Neural Networks (CNNs) and a web application built with Streamlit.

## Project Overview

This project aims to create an interactive web application that can classify bone fracture images with high accuracy. The model architecture leverages the power of CNNs to extract features and make predictions.

## Model Details

### Architecture

The model uses the following layers:
- **Conv2D**: Applies convolutional filters to extract features from input images.
- **BatchNormalization**: Normalizes layer activations to stabilize training.
- **MaxPooling2D**: Downsamples feature maps, preserving dominant features.
- **Dropout**: Regularizes the model by randomly dropping neurons during training.
- **Dense**: Fully connected layers for classification based on extracted features.

### Optimizer
- **Adam**

### Loss Function
- **Binary Crossentropy**

### Metrics
- **Accuracy**
- **Specificity at Sensitivity 0.5**
- **AUC**

### Special Features
- **Feature Hierarchies**: CNN builds progressively complex features from low-level to high-level representations.
- **Weight Sharing**: Parameters in convolutional layers are shared across space, reducing model size.
- **Translation Invariance**: Achieved through pooling layers, allowing detection of features regardless of position.
- **Adaptive Learning Rates**: Adam optimizer adjusts learning rates for each parameter individually.
- **Regularization**: Dropout adds noise to prevent overfitting, improving model generalization.

## Streamlit Web Application

The web application is built using Streamlit and allows users to upload an image, which is then classified by the model.

### Navigation

- **Model Details**: Information about the model architecture and special features.
- **Author**: Information about the author.
- **Model Prediction**: Upload an image for classification.
