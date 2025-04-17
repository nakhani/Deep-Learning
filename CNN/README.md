# Accuracy Comparison: MLP vs CNN+MLP

This repository compares several datasets using two approaches:
1. **MLP (Machine Learning)** 
2. **CNN + MLP (Deep Learning)** 

## Accuracy Table

| **Dataset**       | **MLP (Machine Learning)** 😭 | **CNN + MLP (Deep Learning)** 😍 |
|--------------------|-------------------------------|----------------------------------|
| **MNIST**         | 97%                           | **98%**                         |
| **Fashion MNIST** | 87%                           | **90%**                         |
| **CIFAR-10**      | 36%                           | **70%**                         |
| **CIFAR-100**     | 15%                           | **49%**                         |

## Datasets Overview

### 1. **MNIST** 🖊️
- **Description**: A dataset containing **70,000 grayscale images** of handwritten digits (0-9), with 60,000 for training and 10,000 for testing.
- **Image Size**: **28x28 pixels**, representing a single digit in the center.
- **Purpose**: Ideal for testing basic machine learning and deep learning models.
- **Complexity**: Simple, as the data is clean and has no background noise.

---

### 2. **Fashion MNIST** 👗👞
- **Description**: A modern dataset with **70,000 grayscale images** of fashion items (clothing, shoes, bags, etc.), including 60,000 for training and 10,000 for testing.
- **Image Size**: **28x28 pixels**, where each image represents one of 10 categories like T-shirts, trousers, or sneakers.
- **Purpose**: Provides a slightly more challenging benchmark than MNIST.
- **Complexity**: Moderate, with overlapping visual similarities between some classes.

---

### 3. **CIFAR-10** 🖼️
- **Description**: A dataset of **60,000 color images** categorized into 10 classes such as airplanes, cars, cats, and dogs. Divided into 50,000 for training and 10,000 for testing.
- **Image Size**: **32x32 pixels**, with 3 color channels (RGB).
- **Purpose**: Great for testing image recognition models on low-resolution real-world images.
- **Complexity**: Challenging, due to diverse classes and background noise.

---

### 4. **CIFAR-100** 🌍
- **Description**: An extension of CIFAR-10 with **60,000 color images** spread across **100 fine-grained classes**, such as "flowers," "vehicles," and "reptiles." Each class has 500 training images and 100 testing images.
- **Image Size**: **32x32 pixels**, with 3 color channels (RGB).
- **Purpose**: Introduced for fine-grained classification tasks requiring better feature extraction.
- **Complexity**: Significantly harder than CIFAR-10 due to the increased number of classes.

---

## Summary
- Deep Learning (CNN + MLP) outperforms traditional MLP across all datasets, especially for complex datasets like CIFAR-10 and CIFAR-100.
- This demonstrates the strength of CNNs in learning hierarchical features and handling visual complexities.


---

## How to Run the Code
1. Clone the repository:

   ```
   https://github.com/nakhani/Deep-Learning/tree/cae1786a1cf596db0fddf738a728c07e9f306f32/Face%20recognition
   ```

2. Navigate to the directory:

   ```
   CNN
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the project:
  
   ```
   jupyter notebook cifar10_cnn+mlp.ipynb  # For training Cifar10 Dataset model with CNN + MLP
   jupyter notebook cifar10_MLP.ipynb    # For training Cifar10 Dataset model with MLP
   jupyter notebook cifar100_cnn+mlp.ipynb    # For training Cifar100 Dataset model with CNN + MLP
   jupyter notebook cifar100_MLP.ipynb    # For training Cifar100 Dataset model with MLP
   jupyter notebook fashion_mnist_cnn+mlp.ipynb    # For training Fashion Mnist Dataset model with CNN + MLP
   jupyter notebook fashion_mnist_MLP.ipynb    # For training Fashion Mnist Dataset model with MLP
   jupyter notebook mnist_cnn+mlp.ipynb    # For training Mnist Dataset model with CNN + MLP
   jupyter notebook Mnist_MLP.ipynb    # For training Mnist Dataset model with MLP
   ```

---
## Technologies Used
- Python 3
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Deepface
- scikit-learn
