import gradio as gr # Used to build the GUI
from PIL import Image # Converts the user uploaded images into RGB format and resizes them to the correct dimensions.
import numpy as np # Converts image data to a NumPy array so it can be passed into the model
from keras.models import load_model # Loads the .h5 pre-trained model 

# Loads the model 
model = load_model("model/skin_lesion_model.h5")

# Class labels used by the model
class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# This function is called when a user uploads an image, it preprocesses the image, feeds it into the model,
# and returns the predicted class and confidence scores
def classify_from_upload(uploaded_img):
    try:
        img = uploaded_img.convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)

        # Returns a seven element lost of probabilities
        preds = model.predict(img_array)[0]
        top_index = np.argmax(preds) # finds the highest confidence scoreand saves it in top_class
        top_class = class_names[top_index]

        # Format confidence scores and returns a dictionary of all class probabilities for the bar chart used in the GUI
        confidence_scores = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
        return f"Prediction: {top_class} with confidence {preds[top_index]:.2%}", confidence_scores
# Error handling
    except Exception as e:
        return "Error", {"Error": str(e)}

# Accepts image inputs and outputs a prediction with labels and a bar chart.
gr.Interface(
    fn=classify_from_upload,
    inputs=gr.Image(type="pil", label="Upload Lesion Image"),
    outputs=[
        gr.Textbox(label="Top Prediction"),
        gr.Label(num_top_classes=7, label="Class Probabilities")
    ],
    title="SkinScanAI - Lesion Risk Classifier",
    description="Upload an image of a mole or skin lesion to receive a predicted risk category."
).launch()
