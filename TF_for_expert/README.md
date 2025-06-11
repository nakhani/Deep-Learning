# TensorFlow for Experts

This repository contains advanced implementations of deep learning classification projects using **TensorFlow's expert mode**. Each project follows best practices for creating **custom models**, writing **forward passes imperatively**, and implementing **custom layers, activations, and training loops**.

## Projects

### 1. MNIST
-  write MNIST classification project using TensorFlow for experts mode.
-  Create a class for my model.
-  Write the forward pass imperatively, including custom layers, activations, and training loops.

### 2. CIFAR-10
-  write CIFAR-10 classification project using TensorFlow for experts mode.
-  Create a class for my model.
-  Write the forward pass imperatively, including custom layers, activations, and training loops.

### 3. Titanic
-  write Titanic classification project using TensorFlow for experts mode.
-  Create a class for my model.
-  Write the forward pass imperatively, including custom layers, activations, and training loops.

---

# Model Performance

| Name    | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|---------|------------|---------------|-----------|--------------|
| MNIST   | 0.005     | 0.99         | 0.07    | 0.98        |
| CIFAR-10 | 0.57    | 0.79         | 0.97     | 0.68        |
| Titanic | 0.38      | 0.84         | 0.32     | 0.86       |

---
## How to Run the Code
1. Clone the repository:

   ```
   https://github.com/nakhani/Deep-Learning/tree/59270ea763a97413212abd5bfd71769ee35a2995/TF_for_expert
   ```

2. Navigate to the directory:

   ```
   TF_for_expert
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the project:
  
   ```
   jupyter notebook Mnist.ipynb  # For classifying Mnist Dataset with TensorFlow's expert mode
   jupyter notebook Cifar10.ipynb    # For classifying Cifar10 Dataset with TensorFlow's expert mode
   jupyter notebook Titanic.ipynb  # For classifying Titanic Dataset with TensorFlow's expert mode
   
   ```
   
---
## Technologies Used
- Python 3
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib

