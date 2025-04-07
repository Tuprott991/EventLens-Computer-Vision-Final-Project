import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from imblearn.over_sampling import SMOTE
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
            self.encoded_labels, self.album_ids = self.apply_smote_multioutput(self.encoded_labels, self.album_ids)

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

    def apply_smote_multioutput(self, encoded_labels, album_ids, target_count=1200):
        encoded_labels = np.array(encoded_labels)
        album_ids = np.array(album_ids)
        num_labels = encoded_labels.shape[1]

        smote_features = np.random.rand(len(encoded_labels), 10)  # Dummy feature vectors
        X_resampled_list = []
        album_ids_resampled_list = []

        for i in range(num_labels):
            y = encoded_labels[:, i]
            current_count = np.sum(y)
            if current_count < 6 or current_count >= target_count:
                continue

            sampling_strategy = min(1.0, target_count / current_count - 1e-6)
            sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42)

            try:
                X_res, y_res = sm.fit_resample(smote_features, y)
            except ValueError:
                continue

            num_new = len(X_res) - len(smote_features)
            if num_new <= 0:
                continue

            # Duplicate label rows & album_ids
            new_labels = np.tile(encoded_labels[y == 1][0], (num_new, 1))
            new_album_ids = np.random.choice(album_ids[y == 1], num_new)

            X_resampled_list.append(new_labels)
            album_ids_resampled_list.extend(new_album_ids)

        if len(X_resampled_list) > 0:
            new_labels_all = np.vstack(X_resampled_list)
            new_album_ids_all = np.array(album_ids_resampled_list)
            encoded_labels = np.vstack([encoded_labels, new_labels_all])
            album_ids = np.concatenate([album_ids, new_album_ids_all])

        return encoded_labels, album_ids

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
