import os
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from dataset import AlbumEventDataset
from model_arch import EventLens, visualize_attention

IMAGE_ROOT = 'CUFED5_Album/'
NUM_LABELS = 23
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_labels = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity', 'CasualFamilyGather',
                'Christmas', 'Cruise', 'Graduation', 'GroupActivity', 'Halloween',
                'Museum', 'NatureTrip', 'PersonalArtActivity', 'PersonalMusicActivity',
                'PersonalSports', 'Protest', 'ReligiousActivity', 'Show', 'Sports',
                'ThemePark', 'UrbanTrip', 'Wedding', 'Zoo']

model = EventLens(num_labels=NUM_LABELS)
model.load_state_dict(torch.load('eventlens_best_model_convNeXT_v2.pth'))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize ảnh về kích thước phù hợp với ConvNeXt
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Mean và Std phù hợp với ConvNeXt
                         std=[0.5, 0.5, 0.5])
])

def load_album_images(album_path, transform=None):
    images = []
    files = sorted(os.listdir(album_path))[:20]
    for img_file in files:
        img_path = os.path.join(album_path, img_file)
        img = Image.open(img_path).convert('RGB')
        if transform:
            img = transform(img)
        images.append(img)

    while len(images) < 20:
        images.append(torch.zeros_like(images[0]))

    return torch.stack(images)

def infer_album(album_path):
    album_images = load_album_images(album_path, transform)
    album_images = album_images.to(DEVICE)

    with torch.no_grad():
        logits, attention_weights = model(album_images.unsqueeze(0))
        outputs = torch.sigmoid(logits).cpu().numpy()

    album_labels = []
    for i, label in enumerate(model_labels):
        if outputs[0][i] > 0.7:
            album_labels.append(label)

    rearranged_images = visualize_attention(album_images, attention_weights)

    return album_labels, rearranged_images