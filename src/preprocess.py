from PIL import Image # Opens and converts image to RGB 
import numpy as np # Converts image data into a numerical array 
import os # Ensures we don't need to hardcode a filepath when inputting images

# This function is called by predict.py to get an image ready for model input.
def preprocess_image(image_id, target_size=(224, 224), image_dir="all_images/"):
    img_path = os.path.join(image_dir, f"{image_id}.jpg") # Loads dataset images based on their ID 
    img = Image.open(img_path).convert("RGB").resize(target_size) # loads the image, converts to RGB, resizes to 224x224
    img_array = np.asarray(img) / 255.0 # Converts the image into a NumPy array 
    return img_array # returns the processed image so it can be expanded into a batch in predict.py
