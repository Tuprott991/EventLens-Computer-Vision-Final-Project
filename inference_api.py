import gradio as gr
import requests
import os
import base64
import cv2

import numpy as np
API_URL = "http://103.78.3.25:8000/predict/"  # Change to real API endpoint 

def decode_numpy_image(img_b64):
    # If your API returns the image as a base64 string
    img_bytes = base64.b64decode(img_b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def predict_album_api(uploaded_files):
    files = []
    for file in uploaded_files:
        files.append(('files', (file.name, open(file.name, 'rb'), 'image/jpeg')))
    try:
        response = requests.post(API_URL, files=files)
        result = response.json()
        album_labels = result.get("labels", {})
        # If your API returns attention images, handle them here
        rearranged_images = result.get("rearranged_images", [])
        # Only return the first image if it's a list of images
       # If rearranged_images is a list of base64-encoded images
        if isinstance(rearranged_images, list) and len(rearranged_images) > 0:
            # If it's a numpy array encoded as base64 string
            image_to_show = decode_numpy_image(rearranged_images[0])
        elif isinstance(rearranged_images, str):
            image_to_show = decode_numpy_image(rearranged_images)
        else:
            image_to_show = None

        return album_labels, image_to_show

    finally:
        # Clean up temp files if needed
        pass

iface = gr.Interface(
    fn=predict_album_api,
    inputs=gr.File(file_types=["image"], file_count="multiple", label="Upload Album Images"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predicted Labels"),
        gr.Image(type="numpy", label="Attention Rearranged Images")
    ],
    title="Album Event Classification (API)",
    description="Upload multiple album images to classify the event and visualize attention."
)

if __name__ == "__main__":
    iface.launch(share=True, server_name="0.0.0.0", server_port=7860)