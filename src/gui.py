import gradio as gr  # Used to build the GUI
from PIL import Image  # Converts the user uploaded images into RGB format and resizes them to the correct dimensions.
import numpy as np  # Converts image data to a NumPy array so it can be passed into the model
from keras.models import load_model  # Loads the .h5 pre-trained model

# Loads the model 
model = load_model("model/skin_lesion_model.h5")

# Class labels and full readable names
class_labels = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

# Educational links for each lesion
info_links = {
    "akiec": "https://www.mayoclinic.org/diseases-conditions/actinic-keratosis/symptoms-causes/syc-20354969",
    "bcc": "https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354187",
    "bkl": "https://www.yalemedicine.org/conditions/seborrheic-keratosis",
    "df": "https://dermnetnz.org/topics/dermatofibroma",
    "mel": "https://www.yalemedicine.org/conditions/melanoma-treatment-options",
    "nv": "https://www.yalemedicine.org/conditions/melanocytic-nevi-moles",
    "vasc": "https://www.yalemedicine.org/clinical-keywords/vascular-lesions"
}

# This function is called when a user uploads an image, it preprocesses the image, feeds it into the model,
# and returns the predicted class and confidence scores
def classify_from_upload(uploaded_img):
    try:
        img = uploaded_img.convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)

        # Returns a seven element list of probabilities
        preds = model.predict(img_array)[0]
        top_index = np.argmax(preds)  # Finds the highest confidence score and saves it in top_class

        short_label = list(class_labels.keys())[top_index]
        full_name = class_labels[short_label]
        link = info_links.get(short_label, "#")

        # Create markdown string for prediction
        prediction_text = f"{full_name} with {preds[top_index]:.2%} confidence "

        # Format confidence scores for bar chart
        confidence_scores = {
            class_labels[list(class_labels.keys())[i]]: float(preds[i])
            for i in range(len(preds))
        }

        # Return prediction text, confidence bar, and learn more button
        return prediction_text, confidence_scores, f"[Learn more about {full_name}]({link})"

    except Exception as e:
        return "Error", {"Error": str(e)}, ""

# Custom CSS 
custom_css = """
body {
    background-color: #d6e6f2; /* light blue background */
    color: #000000; /* default text color black */
}

h1 {
    color: white !important;
    font-size: 45px;
    font-family: "Avantgarde", "TeX Gyre Adventor", "URW Gothic L", sans-serif;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    font-weight: bold;
}

h2, h3, label {
    color: #1c2833; /* dark blue */
}

.gr-box, .gr-textbox, .gr-label {
    background-color: #ffffff !important; /* white boxes */
    color: #2e4053 !important; /* black text */
}

.gr-button {
    background-color: #aed6f1 !important; /* light blue buttons */
    color: #1c2833 !important;
    font-weight: bold;
}

.gr-button:hover {
    background-color: #d6eaf8 !important;
}
"""

# Accepts image inputs and outputs a prediction 
gr.Interface(
    fn=classify_from_upload,
    inputs=gr.Image(type="pil", label="Upload Lesion Image"),
    outputs=[
        gr.Textbox(label="Top Prediction"),
        gr.Label(num_top_classes=7, label="Class Probabilities"),
        gr.Markdown(label="Learn More")
    ],
    title="Melanoma Risk Classifier",
    allow_flagging="never",
    description="Upload an image of a mole or skin lesion to receive a predicted risk category.",
    theme=gr.themes.Soft(primary_hue="sky", secondary_hue="slate"),
    css=custom_css
).launch(share=True)


