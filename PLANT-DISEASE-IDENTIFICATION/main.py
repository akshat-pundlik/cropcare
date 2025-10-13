import streamlit as st
import tensorflow as tf
import numpy as np
import os # <--- ADDED for robust path handling
from PIL import Image

# --- CONFIGURATION & ROBUST PATH HANDLING ---

# Get the absolute directory of the current script (main.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource # Caches the model so it only loads ONCE (crucial for Streamlit)
def load_disease_model():
    # Model loading using robust path and stable .h5 format (RECOMMENDED FIX)
    model_filename = "plant_disease_model.h5" # <--- Use the stable .h5 format
    model_path = os.path.join(SCRIPT_DIR, model_filename)
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        # If the model fails to load, display a clear error message
        st.error(f"FATAL ERROR: Could not load the model. Please check file name/path: {model_path}")
        st.stop()
        
# Load the model once at startup
model = load_disease_model() 

# --- FUNCTIONS ---

def model_prediction(test_image):
    # The model is already loaded and cached above, so we don't load it again here.
    # model = tf.keras.models.load_model("trained_plant_disease_model.keras") 

    # tf.keras.preprocessing.image.load_img can handle the file object from st.file_uploader
    image = tf.keras.utils.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr]) # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) # return index of max element

# --- SIDEBAR & INITIAL DISPLAY ---

# Sidebar
st.sidebar.title("AgriSens")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])

# Load and display initial image (using robust path)
try:
    img_filename = "Diseases.png"
    img_path = os.path.join(SCRIPT_DIR, img_filename)
    img = Image.open(img_path)
    st.image(img)
except FileNotFoundError:
    st.warning("Initial image 'Diseases.png' not found. Please upload it.")

# --- MAIN LOGIC ---

# Main Page
if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION", unsafe_allow_html=True)
    
# Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("DISEASE RECOGNITION")
    
    # st.file_uploader returns a file object or None
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        if(st.button("Show Image")):
            # Streamlit displays the file object correctly
            st.image(test_image, width=400) # Use a specific width instead of width=4

        # Predict button
        if(st.button("Predict")):
            st.snow()
            st.write("Our Prediction")
            
            # Predict only if a file is uploaded
            try:
                result_index = model_prediction(test_image)
            except Exception as e:
                st.error(f"Prediction failed. Error: {e}")
                st.stop()
                
            # Reading Labels
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
