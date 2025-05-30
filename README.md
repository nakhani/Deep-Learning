# Deep Learning Repository

This repository contains various deep learning projects covering **face recognition, convolutional neural networks (CNNs), transfer learning, object detection, and optical character recognition (OCR)**.

## 📂 Projects Overview
### 1️⃣ **Face Recognition**
- Convert images to feature vectors using **DeepFace**.
- Save dataset as `.csv`, `.json`, or `.npy`.
- Train a **Multi-Layer Perceptron (MLP)** model.
- Report **train & evaluation loss**.

### 2️⃣ **CNN Accuracy Benchmark**
- Compare accuracy on different datasets: **MNIST, Fashion MNIST, CIFAR-10, CIFAR-100**.
- Complete the accuracy table for **MLP vs. CNN + MLP**.

### 3️⃣ **5 Animals Classification**
- Train a **CNN** on the 5Animals dataset.
- Plot **confusion matrix**.
- Write an **inference code** to predict emojis 🐘🦒🐼🐶🐱.

### 4️⃣ **17 Flowers Classification**
- Train a **CNN** on **17Flowers dataset**.
- Apply **data augmentation**.
- Compare training results.
- Connect the trained model to a **Telegram bot**.

### 5️⃣ **Transfer Learning for Image Classification**
- Retrain models with **Transfer Learning**.
- Compare results on **5 Animals, 17 Flowers, 7-7 Faces**.
- Evaluate performance improvements.

### 6️⃣ **Akhund & Human Face Recognition**
- Train CNN on the **Akhund-and-Human dataset**.
- Apply **face alignment tools**.
- Connect to **wandb** for logging.
- Build a **Telegram bot** for real-time classification.

### 7️⃣ **Age Prediction**
- Download **UTK Face dataset**.
- Analyze **23,000 images**.
- Apply **data augmentation** (horizontal flip).
- Train a **CNN model using Transfer Learning**.
- Predict age of a person from their face.

### 8️⃣ **Home Price Prediction**
- Use **CNNs for regression**.
- Input **four images** of a home to the model.
- Predict **price based on home images**.

### 9️⃣ **YOLO: Persian License Plate Recognition**
- Collect various **Persian license plate images**.
- Create labels using **Roboflow**.
- Prepare dataset for **YOLOv8 training**.
- Implement inference for **license plate detection & cropping**.

### 🔟 **OCR & Deep Text Recognition**
#### **EasyOCR**
- Install **EasyOCR** package.
- Run inference on:
  - **Latin handwriting text**.
  - **Persian handwriting text**.
  - **Latin license plate images**.
  - **Persian license plate images**.

#### **Deep Text Recognition Benchmark (DTRB)**
- Clone **DTRB repository**.
- Choose **pretrained model** & download it via `gdown`.
- Annotate **64 detected license plate images** from YOLO.

---

> ⚠️**NOTE**: You can navigate to each directory independently, check its README file, and follow the instructions and steps provided to run the project of your choice.
