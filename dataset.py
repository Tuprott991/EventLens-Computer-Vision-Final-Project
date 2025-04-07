import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from collections import Counter


class AlbumEventDataset(Dataset):
    def __init__(self, json_path, image_root, transform=None, max_images=10, oversampling=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.album_ids = list(self.data.keys())
        self.labels = list(self.data.values())
        self.image_root = image_root
        self.transform = transform
        self.max_images = max_images
        self.oversampling = oversampling

        self.label_binarizer = MultiLabelBinarizer()
        self.label_binarizer.fit(self.labels)
        self.encoded_labels = self.label_binarizer.transform(self.labels)

        # Print label frequencies before applying oversampling
        self.print_label_frequencies()

        if self.oversampling:
            self.encoded_labels, self.album_ids = self.apply_random_oversampling(self.encoded_labels, self.album_ids)

        # Print label frequencies after applying oversampling
        self.print_label_frequencies()

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

    def apply_random_oversampling(self, encoded_labels, album_ids):
        """
        Applies random oversampling to balance the dataset by duplicating labels.
        """
        label_counts = Counter(map(tuple, encoded_labels))  # Count occurrences of each label
        max_count = max(label_counts.values())  # Get the maximum count of any label

        resampled_labels = []
        resampled_album_ids = []

        for idx, label in enumerate(encoded_labels):
            resampled_labels.append(label)
            resampled_album_ids.append(album_ids[idx])

            # Add duplicate samples for underrepresented labels
            while label_counts[tuple(label)] < max_count:
                resampled_labels.append(label)
                resampled_album_ids.append(album_ids[idx])
                label_counts[tuple(label)] += 1

        return np.array(resampled_labels), np.array(resampled_album_ids)

    def print_label_frequencies(self):
        """ Prints the frequency of each label in the dataset. """
        label_counts = {label: 0 for label in self.label_binarizer.classes_}
        for encoded_label in self.encoded_labels:
            for i, count in enumerate(encoded_label):
                if count > 0:
                    label_counts[self.label_binarizer.classes_[i]] += 1
        print("\nLabel Frequencies:")
        for label, count in label_counts.items():
            print(f"Label: {label}, Frequency: {count}")


# Testing the dataset with oversampling enabled
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
        max_images=32,
        oversampling=True  # Enable random oversampling
    )
