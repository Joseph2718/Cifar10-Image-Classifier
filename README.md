# CIFAR-10 Image Classifier

This project is a web application for image classification on the CIFAR-10 dataset. It uses a Convolutional Neural Network (CNN) model trained on the CIFAR-10 dataset to classify images into one of 10 classes: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', and 'truck'. The web interface for the application is built using Streamlit.

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

