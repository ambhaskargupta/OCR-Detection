# OCR-Detection

# Optical Character Recognition (OCR) System

## Introduction

This project focuses on building an Optical Character Recognition (OCR) system using Convolutional Neural Networks (CNNs) and Transfer Learning. The goal is to develop a model that can accurately recognize and classify characters in images.

## Key Components and Steps

**Dataset:** The project utilizes the "Standard OCR Dataset" containing a variety of characters for training and testing.

**CNN Model:** A custom CNN model is constructed using layers like Conv2D, MaxPooling2D, Flatten, and Dense.

**Transfer Learning:** To enhance performance, transfer learning is employed using the pre-trained VGG16 model.

**Evaluation:** The performance of both models is evaluated using metrics like loss and accuracy.

**Real-World Application:** The project demonstrates a real-world application by detecting and recognizing text in an image.

## Features

This OCR system boasts the following features:

**Core Functionality:**

* **Character Recognition:** Accurately recognizes and classifies individual characters, including digits (0-9) and uppercase letters (A-Z).
* **Custom CNN Model:** Utilizes a custom-built Convolutional Neural Network (CNN) for feature extraction and character classification.
* **Transfer Learning with VGG16:** Leverages the pre-trained VGG16 model for enhanced performance and faster training.
* **Image Preprocessing:** Incorporates preprocessing steps like grayscale conversion, binary thresholding, and dilation for improved input quality.
* **Text Region Detection:** Detects text regions within images to isolate and process specific areas containing characters.
* **Prediction and Decoding:** Predicts characters in detected regions and decodes the predictions to obtain recognized text.
* **Real-World Application:** Demonstrates practical use cases for text detection and recognition in real-world images.

**Additional Features:**

* **Data Visualization:** Provides visualizations of training history and model performance metrics (loss, accuracy).
* **Model Evaluation:** Evaluates the performance of both the custom CNN and transfer learning models on a test dataset.
* **Classification Report:** Generates a classification report with detailed performance metrics for each character class.
* **Bounding Box Visualization:** Optionally draws bounding boxes around detected words in the input image.

## Results

- The project achieved 96% Accuracy on the test dataset.
- Visualizations of training history and model performance are included in the notebook.

## Conclusion

This project demonstrates the effective use of CNNs and transfer learning for OCR. It provides a foundation for further development in text extraction and automation.

## Potential Improvements

- Explore different CNN architectures and hyperparameter optimization.
- Implement data augmentation techniques to improve model robustness.
- Integrate language models for enhanced word-level recognition.


