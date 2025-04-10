# MNIST Digit Classification – CS361 Machine Learning Project (IIT Guwahati)

This repository contains our implementation and comparison of **four different approaches** for handwritten digit classification using the **MNIST dataset**. This project was completed as part of the **CS361 Machine Learning course** at **IIT Guwahati**.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Models Implemented](#models-implemented)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- 
---

## 🧠 Overview

MNIST is a classic benchmark dataset of handwritten digits (0–9), with 60,000 training and 10,000 test grayscale images of size 28×28 pixels.

We explore and implement **four classification approaches** from scratch and/or using frameworks, comparing performance, accuracy, and runtime:

1. **Convolutional Neural Network (CNN)**
2. **K-Nearest Neighbors (KNN)**
3. **Naive Bayesian Classifier (with geometric mean correction)**
4. **Logistic Regression**

---

## 🧪 Models Implemented

### 1. 🧱 CNN (Convolutional Neural Network)
- Implemented using **TensorFlow/Keras**
- Achieves >98% accuracy
- Includes data augmentation, batch normalization, dropout

### 2. 🔍 KNN (from-scratch)
- Includes **Euclidean** and **Cosine similarity**
- Vectorized implementation using NumPy
- Also includes visualization and confusion matrix
- Slow for full dataset, but optimized with batching/vectorization

### 3. 📊 Naive Bayesian (Geometric Mean Variant)
- Inspired by the paper: _"A Novel Naive Bayesian Approach..."_
- Avoids underflow using geometric mean instead of product of probabilities
- Works on full MNIST with custom Gaussian assumption per pixel

### 4. ➗ Logistic Regression
- Used as a baseline linear classifier
- Surprisingly effective with proper regularization

---

## 📈 Results Summary

| Model                | Accuracy (≈) | Runtime | Notes                             |
|---------------------|--------------|---------|-----------------------------------|
| CNN                 | 98–99%       | Fast    | Best performer, deep learning     |
| KNN (Cosine)        | 94–96%       | Slow    | Accurate but slow for full data   |
| Bayesian (Geometric)| 89–92%       | Medium  | Custom math-based formulation     |
| Logistic Regression | 91–93%       | Fast    | Solid baseline for comparison     |

---

## 📁 Project Structure

```bash
.
├── cnn_digit.ipynb                 # CNN implementation notebook
├── knn_digit.ipynb                 # KNN from-scratch + visualization
├── bayesian_classifier.ipynb       # Naive Bayesian classifier with geometric mean
├── logistic_regression.ipynb       # Logistic Regression with scikit-learn
├── README.md                    
