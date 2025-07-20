'''
predict.py handles the more AI-focused logic, calls gui.py when a user uploads an image, 
uses preprocess.py to prepare the image, and uses the .h5 model to generate diagnosis prediction.
'''
import numpy as np # Used to reshape the input images into a batch format (multiple NumPy arrays organized and stored together for more efficient processing)
from tensorflow.keras.models import load_model # Loads the pre-trained skin lesion classification model (skin_lesion_model.h5)
from preprocess import preprocess_image # Loads an image from the HAM10000 dataset

# Loads the pre-trained model developed by Syaha
model = load_model("../model/skin_lesion_model.h5")

# Lists the seven class labels given by the dataset metadata. These will be used to translate model output indices into readable predictions.
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

# Predict function that will be called from gui.py when a user uploads an image.
def predict_image(image_id, image_dir="all_images"):
  img = preprocess_image(image_id, image_dir) # loads the image input using preprocess.py and corrects the format.
  img_batch = np.expand_dims(img, axis=0) # wraps a single image into a batch 
  probs = model.predict(img_batch) # feeds the batch into the pre-trained model and returns a seven-element array.
  prediction_index = np.argmax(probs) #find the index with the highest probability corresponding to one of the class labels.
  
  # Dictionary is returned to gui.py 
  return {
      "predicted_label": class_labels[prediction_index],
      "confidence": float(probs[0][prediction_index]),
      "all_confidences": {class_labels[i]: float(probs[0][i]) for i in range(len(class_labels))}
  }
