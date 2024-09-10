# Leukemia Cell Classification using NVIDIA Jetson Nano

## Introduction

This project focuses on classifying and identifying leukaemia cells from normal cells using deep learning techniques. Leveraging the power of the NVIDIA Jetson Nano and ResNet18 architecture, the model aims to distinguish between leukaemia and normal cells with improved accuracy. This solution could contribute to faster and more efficient diagnostics of leukaemia.

## Technologies Used
- **Python**: The core programming language for the project.
- **NVIDIA Jetson Nano**: A compact, powerful computer for running AI inference models at the edge.
- **ResNet18**: A deep learning architecture employed for image classification.
- **Jetson Inference**: NVIDIA’s inference toolkit for deploying the deep learning model.
- **Deep Learning**: Applied to enhance the image classification task.

## Dataset
- [Leukemia Classification Dataset 1](https://www.kaggle.com/datasets/hamzairfan503/leukemia-classification-dataset/data)
- [Leukemia Classification Dataset 2](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification/discussion/293071)

The datasets consist of labelled images, enabling the model to learn to classify leukaemia cells from normal ones.

## Goal
The primary goal of this project is to **improve accuracy** in classifying leukaemia cells to enhance early detection and diagnostic processes.

## Accessories & Resources:
- Jetson Developer Kit 
- Type C power (5V) supply
- Ethernet cable
- HDMI Cable
- Monitor with HDMI cable
- Camera (Logitech C270 HD WEBCAM)
- Keyboard & Mouse (wireless)
- Memory card (more than 32 GB)
- Optional: cooling fan, micro-USB cable(for headless mode)
- Jetson-Inference With Docker File: [https://github.com/dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)
- Pre-trained model: [](https://drive.google.com/drive/folders/1galH0g3vvRG6K12Bl6jAvCBWPM5r3aKI?usp=sharing)



## Setup and Training Instructions
### 1. Preparing for Setup:
- Setup guide for Jetson Nano Developer Tool Kit:
  - Connect SD card to your PC/Laptop.
  - Download the [SD card image](https://developer.nvidia.com/jetson-nano-sd-card-image).
  - Download, install, and launch [Etcher](https://etcher.balena.io/).
  - Format the SD card with [SD Card Formatter](https://www.sdcard.org/downloads/formatter/sd-memory-card-formatter-for-windows-download/).
  - Please Refer [Get Started With Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro) for more information on setup

### 2.  Downloading Jetson Inference with Docker Container:
-  Open Terminal and run:
  ```bash
  git clone --recursive https://github.com/dusty-nv/jetson-inference
```
Wait for the download to complete (may take 10-15 minutes on slow connections).

### 3. 
Here is your README.md file formatted for GitHub:

markdown
Copy code
# Leukemia Cell Classification using NVIDIA Jetson Nano

## Introduction

This project focuses on classifying and identifying leukaemia cells from normal cells using deep learning techniques. Leveraging the power of the NVIDIA Jetson Nano and ResNet18 architecture, the model aims to distinguish between leukaemia and normal cells with improved accuracy. This solution could contribute to faster and more efficient diagnostics of leukemia.

## Technologies Used
- **Python**: The core programming language for the project.
- **NVIDIA Jetson Nano**: A compact, powerful computer for running AI inference models at the edge.
- **ResNet18**: A deep learning architecture employed for image classification.
- **Jetson Inference**: NVIDIA’s inference toolkit for deploying the deep learning model.
- **Deep Learning**: Applied to enhance the image classification task.

## Dataset
- [Leukemia Classification Dataset 1](https://www.kaggle.com/datasets/hamzairfan503/leukemia-classification-dataset/data)
- [Leukemia Classification Dataset 2](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification/discussion/293071)

The datasets consist of labelled images, enabling the model to learn to classify leukaemia cells from normal ones.

## Goal
The primary goal of this project is to **improve accuracy** in classifying leukaemia cells to enhance early detection and diagnostic processes.

## Setup and Training Instructions

### 1. Preparing for Setup:
- Setup guide for Jetson Nano Developer Tool Kit:
  - Connect SD card to your PC/Laptop.
  - Download the SD card image.
  - Download, install, and launch Etcher.
  - Format the SD card with SD Card Formatter.

### 2. Downloading Jetson Inference with Docker Container:
- Open Terminal and run:
  ```bash
  git clone --recursive https://github.com/dusty-nv
  
Wait for the download to complete (it may take 10-15 minutes on slow connections).
### 3. Running the Docker Container:
- Change directory:
  ```bash
  cd jetson-inference
  ```
- Run the Docker container:
  ```bash
  docker/run.sh
  ```
### 4.Download and Setup the Dataset:
- Download the datasets from Kaggle and extract them to:
  ```bash
  Jetson-inference/python/training/classification/data/
  ```
- Create a labels.txt file in the dataset folder, listing the categories to be detected.
### 5. Training the Dataset:
- Start training:
  ```bash
  python3 model_main.py --model-dir=models/Leukemia --batch-size=4 --workers=1 --epochs=100 data/Leukemia
  ```
- Training takes around 12-14 hours and the Jetson Nano may get hot, so avoid touching it during training.
### 6. Export the Model:
- After training, export the model to ONNX format:
  ```bash
  python3 onnx_export.py --model-dir=models/Leukemia
  ```
- Do the same thing with other dataset
### 7. Testing the Model:
- Test the model using:
  ```bash
  imagenet --model=models/Leukemia/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/Leukemia/labels.txt data/Project/Input data/Project/Output
  ```
### 8. Improving Accuracy:
- You can retrain the model by repeating the training steps to further improve accuracy.

### Output Sample:
- Input: ![Screenshot 2024-09-05 215944](https://github.com/user-attachments/assets/be54fcc8-b1e5-43fe-b8fb-bd44f4d41fbe)
- Output: ![Screenshot 2024-09-05 220009](https://github.com/user-attachments/assets/017801d0-8c4a-4b43-812a-4ee1371b0c5d)



