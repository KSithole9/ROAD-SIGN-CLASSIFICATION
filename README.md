# ROAD-SIGN-CLASSIFICATION
GTSRB-GERMAN-TRAFFIC-SIGN (CNN, RESNET50, VGG16)
Traffic Sign Recognition with Convolutional Neural Networks 

 

1. Introduction 

Interest in self-driving cars is growing in this artificial intelligence and machine learning era. Reading and comprehending traffic signs is essential to their safe navigation. This project aims to improve these vehicles' autonomous capabilities by developing a Convolutional Neural Network (CNN)--based Traffic Sign Classifier. We employ various CNN architectures, including custom CNN, transfer learning with ResNet50, and transfer learning with VGG16, to build and evaluate models for classifying traffic signs. 

 

2. Problem statement 

The challenge of accurately classifying road signs poses a complex task in computer vision due to various factors such as variations in lighting, environmental conditions, perspective, and signage degradation.  

Existing methods for road sign classification often encounter accuracy and robustness limitations, which can lead to errors in practical applications like autonomous driving, traffic management, and navigation systems. Addressing these challenges is crucial to ensure the reliable and precise interpretation of road signs for enhanced safety and efficiency on roadways. 

3. Use case digram
   A diagram of a system

Description automatically generated
Mobile cameras are utilized in the traffic sign classification ecosystem to capture pictures of road signs, which serve as crucial input for convolutional neural network models that have been trained. These models use image analysis to predict traffic signs, which helps self-driving cars make well-informed decisions. These predictions are used by autonomous cars to read traffic signs and react to them, improving road safety and efficiency. Moreover, enhanced awareness of traffic signs by human drivers enhances road safety in general. These players work together to create a dynamic system that encourages safer and more effective and efficient transportation. 
4. Data Acquisition and Preprocessing 

ABOUT THE DATASET 

Our Dataset is from Kaggle and consists of cropped images of road signs. Here are the key characteristics of the dataset: 

Source: Kaggle 

Dataset Name: GTSRB (German Traffic Sign Recognition Benchmark) 

Number of Classes: Originally, the dataset contains more than 43 classes of traffic signs. 

Total Number of Images: The dataset comprises more than 50,000 images in total. 
A graph of classes in training data

Description automatically generated
