import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
# Using the specific keras utilities import path for broad compatibility
from tensorflow.keras.utils import load_img, img_to_array

# -------------------- CONFIGURATION --------------------

# Get the absolute directory of the current script for robust path finding
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The model path is now explicitly defined in the project root
MODEL_FILENAME = "trained_plant_disease_model.keras"
model_path = MODEL_FILENAME

# --- Define Class Labels (CRITICAL: Must match training order) ---
# NOTE: This is an incomplete example. YOU MUST complete this list with 
# the EXACT and COMPLETE list of all 38 (or total) class names.
class_names = [
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Apple___scab",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    # ... ADD REMAINING 32+ CLASS NAMES HERE ...
]

# -------------------- 1. Load the model --------------------

def load_disease_model(path):
    """Loads the model, adding specific error handling for diagnosis."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file NOT found at: {path}")
    
    try:
        # Load the model
        model = keras.models.load_model(path)
        return model
    except Exception as e:
        # This error often points to version incompatibility or file corruption.
        raise RuntimeError(f"Failed to load model from {path}. Check TensorFlow version compatibility. Details: {e}")

try:
    print(f"Attempting to load model from: {model_path}")
    model = load_disease_model(model_path)
    model.summary()
    print("\nModel loaded successfully.")
    
except (FileNotFoundError, RuntimeError) as e:
    print(f"FATAL ERROR: {e}")
    exit(1)


# -------------------- 3. Utility: load and preprocess a single image --------------------

def load_and_preprocess(img_path, target_size=(224, 224)):
    """Loads, resizes, normalizes, and expands dimensions of an image."""
    # Check if image file exists before trying to load it
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Test image NOT found at: {img_path}")
        
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # make batch of 1
    return img_array

# -------------------- 4. Predict on a sample image --------------------
# NOTE: YOU MUST REPLACE THIS PATH with an actual, valid image path!
TEST_IMAGE_PATH = "test_images/sample_apple_rot.jpg" # EXAMPLE PATH

if os.path.exists(TEST_IMAGE_PATH):
    try:
        print(f"\nAttempting prediction on: {TEST_IMAGE_PATH}")
        img_arr = load_and_preprocess(TEST_IMAGE_PATH, target_size=(224,224))

        # Perform prediction
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
            print(f"WARNING: Prediction index {pred_class_idx} is out of bounds for the defined class_names list (length {len(class_names)}). You MUST update class_names.")

    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")

else:
    print(f"\nSkipping prediction: Test image not found at {TEST_IMAGE_PATH}")
    print("ACTION REQUIRED: Replace TEST_IMAGE_PATH with a valid image file to test.")
