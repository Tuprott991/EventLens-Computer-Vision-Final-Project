import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import os
import json
from infer import model, load_album_images, visualize_attention, model_labels
import shutil


# --- Define image transformations ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_album(album_folder):
    # Load album images for inference
    # Create list name of images in the folder
    list_name = sorted(os.listdir(album_folder))[:20]  # Limit to 20 images
    album_images = load_album_images(album_folder, transform)
    album_images = album_images.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Perform inference
    with torch.no_grad():
        logits, attention_weights = model(album_images.unsqueeze(0))
        outputs = torch.sigmoid(logits).cpu().numpy()

    # Collect labels with probabilities > 0.7
    album_labels = {}
    for i, label in enumerate(model_labels):
        if outputs[0][i] > 0.7:
            album_labels[label] = float(outputs[0][i])  # Add label with confidence score

    # Rearrange images based on attention weights
    rearranged_images = visualize_attention(attentions=attention_weights, image_names=list_name, image_folder=album_folder)

    return album_labels, rearranged_images

def predict_album_2(uploaded_files):
    # Extract file paths from uploaded files
    album_folder = "temp_album_folder"
    os.makedirs(album_folder, exist_ok=True)

    # print("Uploaded files:" + uploaded_files)

     # Copy uploaded files into the album folder
    for file_path in uploaded_files:
        destination = os.path.join(album_folder, os.path.basename(file_path))
        shutil.copy(file_path, destination)

    # Load album images for inference
    list_name = sorted(os.listdir(album_folder))[:20]  # Limit to 20 images
    album_images = load_album_images(album_folder, transform)
    album_images = album_images.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Perform inference
    with torch.no_grad():
        logits, attention_weights = model(album_images.unsqueeze(0))
        outputs = torch.sigmoid(logits).cpu().numpy()

    # Collect labels with probabilities > 0.7
    album_labels = {}
    for i, label in enumerate(model_labels):
        if outputs[0][i] > 0.7:
            album_labels[label] = float(outputs[0][i])  # Add label with confidence score

    # Rearrange images based on attention weights
    rearranged_images = visualize_attention(attentions=attention_weights, image_names=list_name, image_folder=album_folder)

    # Clean up temporary folder
    for file_name in os.listdir(album_folder):
        os.remove(os.path.join(album_folder, file_name))
    os.rmdir(album_folder)

    return album_labels, rearranged_images

# Update Gradio interface to use predict_album_2
iface = gr.Interface(
    fn=predict_album_2,
    inputs=gr.File(file_types=["image"], file_count="multiple", label="Upload Album Images"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predicted Labels"),
        gr.Image(type="numpy", label="Attention Rearranged Images")
    ],
    title="Album Event Classification",
    description="Upload multiple album images to classify the event and visualize attention."
)

if __name__ == "__main__":
    iface.launch()