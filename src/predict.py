import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_image 

# Loads the pre-trained model devloped by shyra
model = load_model("../model/skin_lesion_model.h5")

# Lists the class labels given by the dataset metadata
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

def predict_image(image_id, image_dir="all_images"):
  img = preprocess_image(image_id, image_dir)
  img_batch = np.expand_dims(img, axis=0)
  probs = model.predict(img_batch)
  prediction_index = np.argmax(probs)
  return {
      "predicted_label": class_labels[prediction_index],
      "confidence": float(probs[0][prediction_index]),
      "all_confidences": {class_labels[i]: float(probs[0][i]) for i in range(len(class_labels))}
  }