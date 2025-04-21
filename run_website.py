import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import os
import json
from infer import model, load_album_images, visualize_attention, model_labels

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

# Gradio interface
iface = gr.Interface(
    fn=predict_album,
    inputs=gr.Textbox(label="Album Folder Path", placeholder="Enter the path to the album folder"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predicted Labels"),
        gr.Image(type="numpy", label="Attention Rearranged Images")
    ],
    title="Album Event Classification",
    description="Enter the path to a folder of album images to classify the events and visualize attention."
)

if __name__ == "__main__":
    iface.launch()