# CIFAR-10 Image Classifier (Streamlit Web App)

This project is a web application for image classification on the CIFAR-10 dataset. It uses a Convolutional Neural Network (CNN) model trained on the CIFAR-10 dataset to classify images into one of 10 classes: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', and 'truck'. The web interface for the application is built using Streamlit.

<img width="1511" alt="Screenshot 2023-08-03 at 8 53 56 PM" src="https://github.com/Joseph2718/Cifar10-Image-Classifier/assets/67449339/0b7e799e-f344-4ea6-abec-0442d106b45a">

## Getting Started

### Prerequisites

To run this project, you will need:

- Python3 (built with 3.9)
- Tensorflow
- Streamlit
- Numpy
- Matplotlib
- Pillow

You can install these packages using pip:

pip install tensorflow streamlit numpy matplotlib pillow

### Running the Application

1. First, train the model using the train.py script. This will save the trained model to a file named cifar10_model.h5:

python train.py

2. Once the model has been trained, you can start the web application using the main.py script:

streamlit run main.py

This will start the Streamlit server and provide a URL where you can access the web application.

## Using the Application

To use the application:

1. Visit the URL provided by Streamlit in your web browser.
2. Upload an image using the 'Upload your image here' button.
3. The application will display the uploaded image and a bar chart showing the model's probabilistic predictions for each class.

## Model

The model used in this application is a Convolutional Neural Network (CNN) with three sets of convolutional layers, each followed by a max pooling layer and a dropout layer. After the convolutional layers, the model has a fully connected layer with 128 units and a final softmax layer for the 10 classes. The model is trained using the Adam optimizer and categorical cross-entropy loss.

## Examples
<img width="521" alt="Screenshot 2023-08-03 at 8 52 22 PM" src="https://github.com/Joseph2718/Cifar10-Image-Classifier/assets/67449339/08d07e2e-cd5b-4b85-b533-9ce9d1aed265">

<img width="514" alt="Screenshot 2023-08-03 at 8 54 34 PM" src="https://github.com/Joseph2718/Cifar10-Image-Classifier/assets/67449339/62564114-eea8-41ce-9448-71e585858bdf">
<img width="565" alt="Screenshot 2023-08-03 at 8 55 30 PM" src="https://github.com/Joseph2718/Cifar10-Image-Classifier/assets/67449339/72410213-154f-4298-b54b-b2a74927dab8">

<img width="498" alt="Screenshot 2023-08-03 at 8 55 01 PM" src="https://github.com/Joseph2718/Cifar10-Image-Classifier/assets/67449339/7fdea8fb-34a7-44ed-a1fe-798b66715b66">
<img width="540" alt="Screenshot 2023-08-03 at 8 56 16 PM" src="https://github.com/Joseph2718/Cifar10-Image-Classifier/assets/67449339/693ab3e6-ca33-4f60-8232-48e72db91459">
<img width="546" alt="Screenshot 2023-08-03 at 8 57 02 PM" src="https://github.com/Joseph2718/Cifar10-Image-Classifier/assets/67449339/f97f1e1e-f91d-433b-b09a-4f2ece7bf1c4">

