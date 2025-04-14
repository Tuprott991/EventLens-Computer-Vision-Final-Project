import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from sklearn.metrics import average_precision_score  # For calculating AP
import numpy as np
from old_model_arch import AlbumEventClassifier, load_model
from dataset import AlbumEventDataset, load_labels, prepare_dataset  

def infer(model, dataloader, device):
    all_predictions = []
    all_labels = []
    model.to(device)  # Move the model to the selected device
    with torch.no_grad():  # No need to calculate gradients during inference
        for album_tensor, labels in dataloader:
            album_tensor = album_tensor.to(device)  # Move tensor to selected device
            outputs = model(album_tensor)  # Model outputs probabilities
            all_predictions.append(outputs.cpu().numpy())  # Move back to CPU
            all_labels.append(labels.cpu().numpy())  # Move labels to CPU as well
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_predictions, all_labels

def calculate_map(predictions, labels):
    # Compute the mAP for multi-label classification
    mAP = 0
    num_classes = labels.shape[1]
    for i in range(num_classes):
        ap = average_precision_score(labels[:, i], predictions[:, i])  # Compute AP for each class
        mAP += ap
    mAP /= num_classes
    return mAP

def calculate_precision(predictions, labels, threshold=0.5):
    predictions = (predictions > threshold).astype(int)  # Binarize predictions
    true_positives = np.sum(predictions * labels, axis=0)  # True positives for each class
    predicted_positives = np.sum(predictions, axis=0)  # Predicted positives for each class
    precision = true_positives / (predicted_positives + 1e-10)  # Avoid division by zero
    return precision


if __name__ == '__main__':
    model_path = 'best_model.pth'  
    json_path = r'dataset/label.json'  
    image_root = r'Eval_dataset/CV_event'  

    # Load model and labels
    model = load_model(model_path)
    labels = load_labels(json_path)

    # Prepare dataset and DataLoader
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = prepare_dataset(json_path, image_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions, true_labels = infer(model, dataloader, device)

    mAP = calculate_map(predictions, true_labels)
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    print(f"Precision: {calculate_precision(predictions, true_labels)}")
