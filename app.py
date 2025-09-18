# app.py
import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np

# Load your saved model
@st.cache_resource  # ensures the model loads only once
def load_model():
    return keras.models.load_model("models/age_gender_model_2.keras")

model = load_model()

# Streamlit UI
st.title("🧑‍🦱 Age & Gender Prediction App")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((200, 200))  # same size as training
    img_array = np.array(img_resized) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    age_pred, gender_pred = model.predict(img_array)
    import numpy as np

    # Process results
    predicted_age = age_pred[0][0]
    predicted_gender = "Male" if gender_pred[0][0] > gender_pred[0][1] else "Female"
    predicted_gender_prob = np.max(gender_pred, axis=1)

    # Show results
    st.subheader("🔎 Prediction Results")
    st.write(f"**Predicted Age:** {predicted_age:.1f} years")
    st.write(f"**Predicted Gender:** {predicted_gender}")
    st.balloons()