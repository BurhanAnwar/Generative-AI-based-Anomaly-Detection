# Generative AI-based Anomaly Detection Project

## Overview
This project implements Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) for image generation and anomaly detection using the MNIST Digits and MNIST Fashion datasets. The code includes exploratory data analysis (EDA), model development, training, visualization, and a VAE-based anomaly detection pipeline.

## Repository Contents
- **Generative AI-based anomaly detection.ipynb**: Jupyter Notebook with the complete implementation of EDA, GANs, VAEs, and anomaly detection.
- **README.md**: This file, providing an overview and instructions for running the code.

## Prerequisites
To run the code, ensure the following dependencies are installed:
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter Notebook

Install the required packages using:
```bash
pip install tensorflow numpy matplotlib jupyter
```
## Datasets
MNIST Digits: 70,000 grayscale images (60,000 training, 10,000 testing) of handwritten digits (0-9), each 28x28 pixels.
MNIST Fashion: 70,000 grayscale images (60,000 training, 10,000 testing) of fashion items across 10 classes (e.g., T-shirt, Sneaker).
**Anomaly Detection Dataset**: Sourced from Kaggle (details in the notebook).

## Project Components
**Exploratory Data Analysis (EDA)**:
  - Visualizes sample images from MNIST Digits and Fashion datasets.
  - Provides dataset statistics (70,000 images, 10 classes each).

**Generative Adversarial Networks (GANs)**:
  - Implements GANs with a generator (dense/convolutional layers) and discriminator.
  - Trains on MNIST Digits and a specific MNIST Fashion class (e.g., sneakers).
  - Visualizes generated images and loss curves.

**Variational Autoencoders (VAEs)**:
  - Implements VAEs with encoder, reparameterization trick, and decoder.
  - Trains on MNIST Digits for image generation and specific digit synthesis (e.g., digit 7, sneakers).
  - Visualizes latent space using PCA/t-SNE.

**Comparison and Analysis**:
  - Compares GANs (sharp images, unstable training) and VAEs (smoother images, stable training, interpretable latent space).

**Anomaly Detection with VAE**:
  - Trains VAE on normal data to compute reconstruction errors.
  - Evaluates anomalies using thresholds, with results visualized via histograms, confusion matrices, and classification reports.

## Results
**GANs**: Generated realistic handwritten digits and sneaker images, with varying training stability.
**VAEs**: Produced smoother images with a structured latent space, enabling controlled generation and anomaly detection.
**Anomaly Detection**: Successfully differentiated normal and anomalous samples using reconstruction error thresholds, applicable to fields like finance and healthcare.

## Future Work:
- Extend models to complex datasets (e.g., CIFAR-10, real-world anomaly datasets).
- Improve GAN training stability using techniques like WGAN or progressive growing.
- Enhance VAE anomaly detection by tuning thresholds and exploring hybrid models.

## Author:
 - M Burhan ud din
