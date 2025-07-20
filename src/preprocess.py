from PIL import Image
import numpy as np
import os

def preprocess_image(image_id, target_size=(224, 224), image_dir="all_images/"):
    img_path = os.path.join(image_dir, f"{image_id}.jpg")
    img = Image.open(img_path).convert("RGB").resize(target_size)
    img_array = np.asarray(img) / 255.0
    return img_array
