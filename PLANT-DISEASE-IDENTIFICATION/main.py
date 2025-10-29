import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
# Using the specific keras utilities import path for broad compatibility
from tensorflow.keras.utils import load_img, img_to_array 

# -------------------- CONFIGURATION --------------------

# Get the absolute directory of the current script for robust path finding
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "plant_disease_model.keras"

# Assume the model is in the same directory as this script.
model_path = os.path.join(SCRIPT_DIR, MODEL_FILENAME)

# --- Define Class Labels (CRITICAL: Must match training order) ---
# NOTE: You MUST update this list with the EXACT and COMPLETE list of
# all disease classes in the same order as your model was trained.
class_names = [
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    # ... add the remaining 35 class names here ...
]

# -------------------- 1. Load the model --------------------

def load_disease_model(path):
    """Loads the model, adding specific error handling for diagnosis."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file NOT found at: {path}")
    
    try:
        # We load using keras.models.load_model as shown in your original file.
        model = keras.models.load_model(path)
        return model
    except Exception as e:
        # This error often points to version incompatibility, even if the file exists.
        raise RuntimeError(f"Failed to load model from {path}. Check TensorFlow version compatibility. Details: {e}")

try:
    print(f"Attempting to load model from: {model_path}")
    model = load_disease_model(model_path)
    model.summary()
    print("\nModel loaded successfully.")
    
except (FileNotFoundError, RuntimeError) as e:
    print(f"FATAL ERROR: {e}")
    # If the file is truly corrupted, loading will fail here.
    exit(1)


# -------------------- 3. Utility: load and preprocess a single image --------------------

def load_and_preprocess(img_path, target_size=(224, 224)):
    """Loads, resizes, normalizes, and expands dimensions of an image."""
    # Use load_img from the imported utility
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # make batch of 1
    return img_array

# -------------------- 4. Predict on a sample image --------------------
# NOTE: Replace this path with an actual test image path available in your environment!
TEST_IMAGE_PATH = "test/Apple___Black_rot/0ef9...jpg" 

if os.path.exists(TEST_IMAGE_PATH):
    print(f"\nAttempting prediction on: {TEST_IMAGE_PATH}")
    img_arr = load_and_preprocess(TEST_IMAGE_PATH, target_size=(224,224))

    pred = model.predict(img_arr)
    pred_class_idx = np.argmax(pred[0])
    pred_confidence = pred[0][pred_class_idx]

    print("Predicted class index:", pred_class_idx)
    print(f"Confidence: {pred_confidence:.4f}")
    
    # Check if class_names has enough entries
    if pred_class_idx < len(class_names):
        pred_class_name = class_names[pred_class_idx]
        print(f"Predicted disease / class: {pred_class_name}")

        # -------------------- 5. Display image + predicted label --------------------
        plt.imshow(load_img(TEST_IMAGE_PATH))
        plt.axis("off")
        plt.title(f"Prediction: {pred_class_name} ({pred_confidence:.2f})")
        plt.show()
    else:
        print(f"WARNING: Prediction index {pred_class_idx} is out of bounds for the defined class_names list (length {len(class_names)}). Update class_names.")

else:
    print(f"\nSkipping prediction: Test image not found at {TEST_IMAGE_PATH}")
    print("Replace TEST_IMAGE_PATH with a valid image file to test.")

# -------------------- 6. (Optional) Batch prediction / Evaluation code from step 6 and 7 --------------------
# The functions for batch prediction and evaluation are not included here as they require 
# a full test directory structure which may not be set up for a single run.
# They can be re-added once the core model loading is confirmed working.
