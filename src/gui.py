import gradio as gr
from PIL import Image
import numpy as np
from keras.models import load_model

# Loads the model 
model = load_model("model/skin_lesion_model.h5")

# Class labels used by the model
class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def classify_from_upload(uploaded_img):
    try:
        # Resize and normalize
        #img = uploaded_img.convert("RGB").resize((224, 224))
        img = uploaded_img.convert("RGB").resize((299, 299))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)

        # Predict
        preds = model.predict(img_array)[0]
        top_index = np.argmax(preds)
        top_class = class_names[top_index]

        # Format confidence scores
        confidence_scores = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

        return f"Prediction: {top_class} with confidence {preds[top_index]:.2%}", confidence_scores

    except Exception as e:
        return "Error", {"Error": str(e)}

# Build GUI
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
