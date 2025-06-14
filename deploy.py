from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import torch
from infer import model, infer_album, load_album_images, visualize_attention, model_labels
from torchvision import transforms
from dataset import AlbumEventDataset
import base64
import cv2

# Function to generate a unique UUID for each request
import uuid
def generate_uuid():
    return str(uuid.uuid4())

def numpy_to_base64(img_np):
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return img_b64

app = FastAPI()
NUM_LABELS = 23
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(files: list[UploadFile] = File(...)):
    uuid = generate_uuid()
    album_folder = uuid
    os.makedirs(album_folder, exist_ok=True)
    file_names = []
    try:
        # Save uploaded files
        for uploaded_file in files:
            file_path = os.path.join(album_folder, uploaded_file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(uploaded_file.file, f)
            file_names.append(uploaded_file.filename)

        # Load and predict
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
        # Rearrange images based on attention weights

        rearranged_images = numpy_to_base64(rearranged_images)  # Convert to base64 string
        
        # Optionally, you can return attention visualization as a file or base64 string
        # rearranged_images = visualize_attention(attentions=attention_weights, image_names=list_name, image_folder=album_folder)

        return JSONResponse(content={"labels": album_labels, "rearranged_images": rearranged_images})

    finally:
        # Clean up
        for file_name in os.listdir(album_folder):
            os.remove(os.path.join(album_folder, file_name))
        os.rmdir(album_folder)



