import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# -------------------- CONFIGURATION --------------------

# Get the absolute directory of the current script (e.g., /path/to/PLANT-DISEASE-IDENTIFICATION/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource # CRUCIAL: Caches the model so it only loads ONCE per deployment.
def load_disease_model():
    """Loads the model using a robust, absolute path."""
    
    model_filename = "plant_disease_model.keras"
    
    # 💡 CORRECT PATH FIX: Use os.path.join to create a reliable absolute path.
    # This path is relative to the directory of this script (main.py).
    model_path = os.path.join(SCRIPT_DIR, model_filename)
    
    try:
        # Check if the file exists before attempting to load
        if not os.path.exists(model_path):
            st.error(f"FATAL ERROR: Model file **NOT FOUND** at: {model_path}")
            st.error("Please ensure 'plant_disease_model.keras' is uploaded to the same directory as this script.")
            st.stop()

        # Attempt to load the model
        model = tf.keras.models.load_model(model_path)
        return model
    
    except Exception as e:
        # Custom error message for file found but could not be loaded
        st.error(f"FATAL ERROR: Could not load the model from **{model_path}**. ")
        st.error("This usually means the **TensorFlow/Keras version** used here is incompatible with the version used to save the model.")
        st.error(f"Underlying System Error: {e}")
        st.stop()

# --- Load the model once at startup ---
# This line will now stop the app if the model cannot be loaded.
model = load_disease_model() 

# -------------------- FUNCTIONS --------------------

def model_prediction(test_image):
    """Performs prediction using the loaded model."""
    
    # Use tf.keras.utils.load_img for reliable loading of file_uploader object
    # The file_uploader object must be passed correctly here
    try:
        # We need to save the uploaded file temporarily to disk 
        # because tf.keras.utils.load_img expects a file path or a string.
        with open("temp_image.png", "wb") as f:
            f.write(test_image.getbuffer())
        
        image = tf.keras.utils.load_img("temp_image.png", target_size=(224, 224)) # Match training size
        
        # Clean up the temporary file
        os.remove("temp_image.png") 
        
        input_arr = tf.keras.utils.img_to_array(image)
        input_arr = np.array([input_arr]) # convert single image to batch
        
        # Rescale the input array
        input_arr = input_arr / 255.0  
        
        predictions = model.predict(input_arr)
        return np.argmax(predictions)
    
    except Exception as e:
        # The original code's prediction block was missing file handling for Streamlit's UploadedFile object.
        raise RuntimeError(f"Error during prediction or image processing: {e}")


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
    st.warning("Initial image 'Diseases.png' not found. Please upload it to the script directory.")

# -------------------- MAIN LOGIC --------------------

if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION", unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")
    
    # Streamlit UploadedFile object
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        
        # Show Image button logic
        if st.button("Show Image"):
            st.image(test_image, width=400)

        # Predict button logic
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            
            try:
                # model_prediction now expects the Streamlit UploadedFile object
                result_index = model_prediction(test_image)
            except RuntimeError as e:
                st.error(f"Prediction failed. {e}")
                # Don't st.stop() here, allow the user to try a new upload.
                
                # --- Disease Labels ---
                # Reading Labels (The labels array must be defined outside the try block for scope)
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
