import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from dataset import AlbumEventDataset
from model_archi2 import EventLens, visualize_attention
import json

from sklearn.metrics import average_precision_score

from sklearn.preprocessing import MultiLabelBinarizer

# --- Config hyperparameters ---
IMAGE_ROOT = 'CUFED5'
OUTPUT_JSON = 'predictions.json'
JSON_PATH = 'dataset/event_type.json'
NUM_LABELS= 23
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_labels = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity', 'CasualFamilyGather',
                'Christmas', 'Cruise', 'Graduation', 'GroupActivity', 'Halloween',
                'Museum', 'NatureTrip', 'PersonalArtActivity', 'PersonalMusicActivity',
                'PersonalSports', 'Protest', 'ReligiousActivity', 'Show', 'Sports',
                'ThemePark', 'UrbanTrip', 'Wedding', 'Zoo']

# --- Load the trained model ---
model = EventLens(num_labels=NUM_LABELS)
model.load_state_dict(torch.load('eventlens_final.pth',map_location=torch.device(DEVICE)))
model = model.to(DEVICE)
model.eval()

# --- Define image transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_label_from_path(path):
    with open(path, 'r') as f:
        data = json.load(f)

    labels = list(data.values())
    # Unique labels and rearranging them
    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(labels)

    # Convert to MultiLabelBinarizer format
    return list(label_binarizer.classes_)

# print(get_label_from_path(JSON_PATH)) 

# Load album images for inference
def load_album_images(album_path, transform=None):
    images = []
    files = sorted(os.listdir(album_path))[:20]  # Limit to 32 images
    for img_file in files:
        img_path = os.path.join(album_path, img_file)
        img = Image.open(img_path).convert('RGB')
        if transform:
            img = transform(img)
        images.append(img)

    # Padding if not enough images
    while len(images) < 20:
        images.append(torch.zeros_like(images[0]))

    return torch.stack(images)  # (N, C, H, W)

# Main
if __name__ == '__main__':
    predictions = {}

    # Iterate through all album directories
    for album_name in sorted(os.listdir(IMAGE_ROOT)):
        album_path = os.path.join(IMAGE_ROOT, album_name)
        if not os.path.isdir(album_path):
            continue

        # Load album images for inference
        album_images = load_album_images(album_path, transform)

        # Move to device
        album_images = album_images.to(DEVICE)

        # Perform inference
        with torch.no_grad():
            logits, _ = model(album_images.unsqueeze(0))  # Get logits and ignore attentions
            outputs = torch.sigmoid(logits).cpu().numpy()

        # Collect labels with probabilities > 0.3
        album_labels = []
        for i, label in enumerate(model_labels):
<<<<<<< HEAD
            if outputs[0][i] > 0.7:
=======
            if outputs[0][i] > 0.75:
>>>>>>> 0623b89404218ac185a10dc200efe337dc8fd494
                album_labels.append(label)

        # Save predictions for the current album
        predictions[album_name] = album_labels

    # Save all predictions to a JSON file
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to {OUTPUT_JSON}")



