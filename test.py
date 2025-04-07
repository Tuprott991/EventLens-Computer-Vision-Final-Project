from model_arch import AlbumEventClassifier
from dataset import AlbumEventDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    # Define the same transformations as in train.py
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the dataset
    dataset = AlbumEventDataset(
        json_path='dataset/CUFED/event_type.json',
        image_root='dataset/CUFED/images',
        transform=transform,
        max_images=32
    )
    
    # Split the dataset (ensure this matches train.py)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=4)
    
    # Initialize the model
    model = AlbumEventClassifier(num_classes=len(dataset.label_binarizer.classes_), aggregator='transformer', max_images=32).cuda()
    
    # Load the trained model weights
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Print true labels (y_true)
    print("Printing true labels (y_true):")
    with torch.no_grad():
        for album_imgs, labels in val_loader:
            labels = labels.cpu().numpy()  # Convert to numpy for easier inspection
            print(labels)