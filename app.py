import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf  # Use your framework (e.g., PyTorch if required)

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.keras")  # Update with your model path
    return model

model = load_model()

# Define the labels
labels = ["Label 1", "Label 2", "Label 3"]  # Replace with your actual labels

# Preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to the input size of your model
    img_array = np.array(img) / 255.0  # Normalize the image (if required by your model)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app layout
st.title("Image Classification App")
st.write("Upload an image, and the model will classify it into one of three labels.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = labels[np.argmax(prediction)]

    # Show results
    st.write(f"**Prediction:** {predicted_label}")
    st.write("Confidence Scores:")
    for i, label in enumerate(labels):
        st.write(f"{label}: {prediction[0][i]:.2f}")
