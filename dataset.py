import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer


class AlbumEventDataset(Dataset):
    def __init__(self, json_path, image_root, transform=None, max_images=10):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.album_ids = list(self.data.keys())
        
        self.labels = list(self.data.values())
        self.image_root = image_root
        self.transform = transform
        self.max_images = max_images

        self.label_binarizer = MultiLabelBinarizer()
        self.label_binarizer.fit(self.labels)
        self.encoded_labels = self.label_binarizer.transform(self.labels)

        # # Print all labels and their corresponding indices
        # print("All Labels and their Indices:")
        # for idx, label in enumerate(self.label_binarizer.classes_):
        #     print(f"Index: {idx}, Label: {label}")

        # # print a sample of album_ids and their labels in both raw and encoded form
        # print("\nSample of Album IDs and their Labels:")
        # for i in range(min(5, len(self.album_ids))):
        #     album_id = self.album_ids[i]
        #     raw_label = self.labels[i]
        #     encoded_label = self.encoded_labels[i]
        #     print(f"Album ID: {album_id}, Raw Label: {raw_label}, Encoded Label: {encoded_label}")

    def __len__(self):
        return len(self.album_ids)

    def __getitem__(self, idx):
        album_id = self.album_ids[idx]
        album_path = os.path.join(self.image_root, album_id)
        label = torch.tensor(self.encoded_labels[idx], dtype=torch.float32)

        images = []
        files = sorted(os.listdir(album_path))[:self.max_images]
        for img_file in files:
            img_path = os.path.join(album_path, img_file)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Padding if not enough images
        while len(images) < self.max_images:
            images.append(torch.zeros_like(images[0]))

        album_tensor = torch.stack(images)  # (N, C, H, W)
        return album_tensor, label
    

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = json.load(f)
    return labels

def prepare_dataset(json_path, image_root, transform=None, max_images=30):
    dataset = AlbumEventDataset(json_path=json_path, image_root=image_root, transform=transform, max_images=max_images)
    return dataset

# Test print the name of album and its labels as raw name

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = AlbumEventDataset(
        json_path='dataset/CUFED/event_type.json',
        image_root='dataset/CUFED/images',
        transform=transform,
        max_images=32
    )