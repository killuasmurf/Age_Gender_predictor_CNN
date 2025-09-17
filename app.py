import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Load trained model
@st.cache_resource
def load_age_gender_model():
    return load_model("age_gender_model.h5")  # replace with your saved model path

model = load_age_gender_model()

# Streamlit UI
st.title("👤 Age & Gender Prediction App")
st.write("Upload a face image (.jpg) to get age and gender predictions.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(200, 200))  # match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    pred = model.predict(img_array)

    # Assuming model outputs [age, gender_probs]
    age_pred = pred[0][0]            # regression output
    gender_prob = pred[1][0]         # probability for female (for example)
    gender_label = "Female" if gender_prob > 0.5 else "Male"

    # Show results
    st.subheader("Results")
    st.write(f"**Predicted Age:** {age_pred:.1f} years")
    st.write(f"**Predicted Gender:** {gender_label} (confidence: {gender_prob:.2f})")
