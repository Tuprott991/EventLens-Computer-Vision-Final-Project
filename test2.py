from old_model_arch import AlbumEventClassifier
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
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)  # Batch size set to 1 for single prediction
    
    # Initialize the model
    model = AlbumEventClassifier(num_classes=len(dataset.label_binarizer.classes_), aggregator='transformer', max_images=32).cuda()
    
    # Load the trained model weights
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Print a single model prediction and true label
    print("Printing a single model prediction and true label:")
    with torch.no_grad():
        for album_imgs, labels in val_loader:  # Assuming dataset returns album_names
            album_imgs, labels = album_imgs.cuda(), labels.cuda()
            outputs = model(album_imgs)
            
            # Convert outputs to probabilities and labels to text
            predicted_probs = outputs.cpu().numpy() 
            
            # Print album name, predicted label, and true label
            print(f"Predicted probs: {predicted_probs}")
            print(f"True Label: {labels}")

            # Print shape of the output and labels
            print(f"Output shape: {predicted_probs.shape}")
            print(f"Labels shape: {labels.shape}")
            break  # Only process the first album