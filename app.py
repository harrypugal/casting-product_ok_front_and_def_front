import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model(r"E:\\sandy\\rjc\\ber.h5")

# Function to preprocess the uploaded image
def preprocess_image(image):
    size = (300, 300) # Make sure this size matches your model's input size
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image = np.asarray(image)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app title
st.title("CNN Model Deployment with Streamlit")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for the model
    processed_image = preprocess_image(image)

    # Make predictions
    prediction = model.predict(processed_image)
    
    if prediction[0][0] > 0.5:
        st.write('This model has def_fron parts')
        st.error('Please note that this model has poor parts in the uploaded image.')
    else:
        st.write('This model has okay parts')
        st.balloons() # Show balloons to indicate the prediction
        
        

    
