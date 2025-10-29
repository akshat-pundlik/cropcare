import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# -------------------- CONFIGURATION --------------------

# Get the absolute directory of the current script for robust path finding
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource # CRUCIAL: Caches the model so it only loads ONCE.
def load_disease_model():
    """Loads the model using a robust, absolute path."""
    model_filename = "plant_disease_model.keras"
    # CORRECT CODE (Defines a string variable)
    model_path = 'PLANT-DISEASE-IDENTIFICATION/plant_disease_model.keras'
    
    try:
        # Load the model using the stable HDF5 format and robust path
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        # Custom error message with the exact path it checked
        st.error(f"FATAL ERROR: Could not load the model from {model_path}. "
                 "Please check file name/path and TensorFlow version.")
        st.stop()

# --- Load the model once at startup ---
model = load_disease_model() 

# -------------------- FUNCTIONS --------------------

def model_prediction(test_image):
    """Performs prediction using the loaded model."""
    # The model object is already available globally from load_disease_model()
    
    # Use tf.keras.utils.load_img for reliable loading of file_uploader object
    image = tf.keras.utils.load_img(test_image, target_size=(224, 224)) # Match training size
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr]) # convert single image to batch
    
    # Rescale the input array (EfficientNet was trained on 0-255 images and expects 0-1)
    # Your training used rescale=1./255, so we apply it here:
    input_arr = input_arr / 255.0 
    
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# -------------------- INITIAL SETUP --------------------

# Sidebar
st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Load and display initial image
try:
    img_filename = "Diseases.png"
    img_path = os.path.join(SCRIPT_DIR, img_filename)
    img = Image.open(img_path)
    st.image(img)
except FileNotFoundError:
    st.warning("Initial image 'Diseases.png' not found. Check upload.")

# -------------------- MAIN LOGIC --------------------

if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION", unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")
    
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, width=400)

        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            
            try:
                result_index = model_prediction(test_image)
            except Exception as e:
                st.error(f"Prediction failed. Error details: {e}")
                st.stop()
                
            # Reading Labels (using the correct list from your previous code)
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']
            
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
    else:
        st.info("Please upload an image file to start prediction.")




