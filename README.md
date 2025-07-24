# SkinScanAI

## A fully functioning website that utilizes a Pre-trained Skin Cancer Detection Model for users to submit images of skin spots to be analyzed.

This was built for our final project for our Artificial Intelligence class that was built with a pre-built dataset that was utilized to train 80% of the pre-trained skin cancer detection model.
The gui was developed using gradio to create a purple and light blue aesthetic.

* Create a custom website using Python for the front and back end.
* Attach the HAM10000 dataset that has seven classifications/labels to test for the main 7 types of skin lesions.
    link to dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T 

* Attach the Pre-trained Skin Cancer Detection Model - syaha/skin_cancer_detection_model.
    link to model: https://huggingface.co/syaha/skin_cancer_detection_model 
predict.py  
  
* Run the dataset on the model to test the accuracy.
* Create a custom interface using Gradio to create a specifc brand and attract users.

  Example of images that the dataset can process:
  
    <img width="254" height="250" alt="image" src="https://github.com/user-attachments/assets/e9ea4e98-75c4-4ed4-b89f-4b290b23654a" /> | 
  <img width="254" height="250" alt="image" src="https://preview.redd.it/gdbulk0vw5y71.jpg?width=3468&format=pjpg&auto=webp&s=4a57363bf42401418175b6aa38fde3fcb994c648" /> |
<img width="254" height="250" alt="image" src="https://github.com/user-attachments/assets/79102805-c521-4e00-ba0e-5c23bcc906ab" />

  
## Mini Tutorial for the implementation of this project.
1. Clone the project
2. Import the dataset and the model
3. In the terminal import gradio, numpy, os, and from tensorflow.keras.models import load_model
4. Test the website by putting in the terminal "python gui.py"
   
