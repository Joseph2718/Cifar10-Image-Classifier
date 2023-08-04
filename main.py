import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def main():
    st.title('Cifar10 Web App: Image Classifier')
    st.write('Upload any image you believe to fit in one of the 10 classes—see if the prediction matches!')

    file = st.file_uploader('Upload your image here', type=['jpeg', 'png'])
    
    if file:
        image = Image.open(file) 
        st.image(image, use_column_width=True)

        # Ensure image is resized, normalized and reshaped in the same way as training images
        resized_image = image.resize((32, 32))
        image_array = np.array(resized_image) / 255  # Normalize pixel values
        image_array = image_array.reshape((1, 32, 32, 3))

        model = tf.keras.models.load_model('cifar10_model.h5')

        predictions = model.predict(image_array)
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        fig, ax = plt.subplots()
        y_pos = np.arange(len(classes))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes)
        ax.invert_yaxis()
        ax.set_xlabel('Probability ')
        ax.set_title('CIFAR10 Predictions')

        st.pyplot(fig)
    else:
        st.text('You have not uploaded an image yet.')

if __name__ == '__main__':
    main()
