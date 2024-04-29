# ROAD-SIGN-CLASSIFICATION
GTSRB-GERMAN-TRAFFIC-SIGN (CNN, RESNET50, VGG16)
Traffic Sign Recognition with Convolutional Neural Networks 

 

1. Introduction 

Interest in self-driving cars is growing in this artificial intelligence and machine learning era. Reading and comprehending traffic signs is essential to their safe navigation. This project aims to improve these vehicles' autonomous capabilities by developing a Convolutional Neural Network (CNN)--based Traffic Sign Classifier. We employ various CNN architectures, including custom CNN, transfer learning with ResNet50, and transfer learning with VGG16, to build and evaluate models for classifying traffic signs. 

 
2. Problem statement 

The challenge of accurately classifying road signs poses a complex task in computer vision due to various factors such as variations in lighting, environmental conditions, perspective, and signage degradation.  

Existing methods for road sign classification often encounter accuracy and robustness limitations, which can lead to errors in practical applications like autonomous driving, traffic management, and navigation systems. Addressing these challenges is crucial to ensure the reliable and precise interpretation of road signs for enhanced safety and efficiency on roadways. 

3. Use case digram
<img width="445" alt="image" src="https://github.com/KSithole9/ROAD-SIGN-CLASSIFICATION/assets/152675019/78323397-466d-48ee-98d4-caf227a34b38">

Mobile cameras are utilized in the traffic sign classification ecosystem to capture pictures of road signs, which serve as crucial input for convolutional neural network models that have been trained. These models use image analysis to predict traffic signs, which helps self-driving cars make well-informed decisions. These predictions are used by autonomous cars to read traffic signs and react to them, improving road safety and efficiency. Moreover, enhanced awareness of traffic signs by human drivers enhances road safety in general. These players work together to create a dynamic system that encourages safer and more effective and efficient transportation. 
4. Data Acquisition and Preprocessing 

ABOUT THE DATASET 

Our Dataset is from Kaggle and consists of cropped images of road signs. Here are the key characteristics of the dataset: 

Source: Kaggle 

Dataset Name: GTSRB (German Traffic Sign Recognition Benchmark) 

Number of Classes: Originally, the dataset contains more than 43 classes of traffic signs. 

Total Number of Images: The dataset comprises more than 50,000 images in total. 

![image](https://github.com/KSithole9/ROAD-SIGN-CLASSIFICATION/assets/152675019/d5758029-b642-4aa6-bb66-2e8b6f2ac849)

Data Preprocessing: 

Class Reduction: While the original dataset contains over 43 classes, we opted to focus on a subset of 24 classes for our specific task. 

Class Imbalance Handling: The dataset was initially imbalanced, with different numbers of samples for each class. To address this, we balanced the dataset by resampling the classes to have 200 samples each. 

![image](https://github.com/KSithole9/ROAD-SIGN-CLASSIFICATION/assets/152675019/f6bc9118-19c2-4769-8144-f1f744dee801)

Dataset Download Link: The dataset can be downloaded from the following lnk: https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign 

By selecting a subset of classes and balancing the dataset, we aimed to create a more manageable and evenly distributed dataset for training our models. 

We obtained the dataset from an archive containing images of traffic signs along with corresponding labels. The dataset includes various classes of traffic signs, such as speed limits, no passing zones, and caution signs. We preprocessed the data by extracting images from the archive, filtering out classes not of interest, and balancing the training data to ensure each class has sufficient representation. 

5. Model Development 

5.1 Model Training and Evaluation 

We trained each model using balanced training data and evaluated their performance on the test dataset. For all models, we employed early stopping and learning rate reduction callbacks to prevent overfitting and enhance convergence. 

5. Custom CNN Model: 

We constructed a custom CNN architecture consisting of convolutional layers followed by max-pooling layers. The model includes three convolutional layers with increasing filter sizes and a fully connected layer with 128 units. The output layer has 24 units corresponding to the 24 traffic sign classes. 

![image](https://github.com/KSithole9/ROAD-SIGN-CLASSIFICATION/assets/152675019/0b218e52-74e9-4733-ab91-a7e498e2ef0c)

![image](https://github.com/KSithole9/ROAD-SIGN-CLASSIFICATION/assets/152675019/369976e0-9fc6-4fc9-89e2-beedfe587e3d)

5.2 Transfer Learning with ResNet50: 


We utilized transfer learning with ResNet50, a deep residual network pre-trained on ImageNet data. We fine-tuned the ResNet50 model by adding custom top layers and training them along with the pre-trained base layers. 

![image](https://github.com/KSithole9/ROAD-SIGN-CLASSIFICATION/assets/152675019/a8a680ec-b393-4986-b130-54db2d0740fd)
![image](https://github.com/KSithole9/ROAD-SIGN-CLASSIFICATION/assets/152675019/abb3c2be-8a6c-4d86-99d3-f5c45f788757)

5.3 Transfer Learning with VGG16: 

Transfer learning with VGG16, another pre-trained model on ImageNet, was employed similarly. We froze most of the VGG16 layers and added custom top layers to adapt the model to our traffic sign classification task. 

![image](https://github.com/KSithole9/ROAD-SIGN-CLASSIFICATION/assets/152675019/174a3a27-9062-4aa8-beab-9e0b550ebdbf)
![image](https://github.com/KSithole9/ROAD-SIGN-CLASSIFICATION/assets/152675019/f828710c-0dd3-426b-9625-f20478479152)


5. Results 

 

The performance of each model was evaluated based on accuracy metrics. Here are the test accuracies achieved by each model: 

VGG16: 84% 

Custom CNN: 91% 

ResNet50: 89% 

A comparison of the accuracies is shown in a bar chart. 

![image](https://github.com/KSithole9/ROAD-SIGN-CLASSIFICATION/assets/152675019/bb7d7aa5-c15a-4dff-9109-b81ce7f7df2e)

6. Individual Image Prediction 

We demonstrated the capability of the ResNet50 model for individual image prediction. An image of a traffic sign was inputted into the model, and the predicted class label was displayed alongside the image. 

![image](https://github.com/KSithole9/ROAD-SIGN-CLASSIFICATION/assets/152675019/c471fda0-7cb2-4a4c-b7f8-72e353d69217)

7. Conclusion 

In conclusion, this project showcases the effectiveness of CNNs, both custom and pre-trained models, for traffic sign recognition. Transfer learning with models like ResNet50 and VGG16 enables us to achieve competitive results with minimal training data. These models can be further optimized and deployed in real-world applications to enhance road safety and support intelligent transportation systems. 

 

8. Reference 

 

X. Mao, S. Hijazi , R. Casas, P. Kaul, R. Kumar, & C. Rowen , Hierarchical CNN for traffic sign recognition. In 2016 IEEE Intelligent 

Autonomous Traffic Sign(ASTR) Detection and Recognition using Deep CNN, Danyah A.Alghmgham , ghazanfar latif , jaafar alghazo , loay alzubaidi ,2019 ScienceDirect 

 A. Kumar, T. Singh, and D. K. Vishwakarma, “Intelligent Transport System: Classification of Traffic Signs Using Deep Neural Networks in Real Time,” Lecture Notes in Mechanical Engineering Advances in Manufacturing and Industrial Engineering, 2021. 

 
