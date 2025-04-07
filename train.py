from model_arch import AlbumEventClassifier
from dataset import AlbumEventDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score

def compute_mAP(outputs, labels):
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return average_precision_score(labels, outputs, average='macro')  # 'macro' for mean over all classes


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Import to Kaggle and input dataset
    dataset = AlbumEventDataset(
        json_path='CUFED/event_type.json',
        image_root='CUFED/images',
        transform=transform,
        max_images=32
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    # initialize the model
    model = AlbumEventClassifier(num_classes=len(dataset.label_binarizer.classes_), aggregator='transformer', max_images=32).cuda()

    # Training settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)   
    num_epochs = 15
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCELoss()
    
    best_val_map = -1
    best_val_loss = float('inf')
    patience = 3
    wait = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
            for album_imgs, labels in tepoch:
                album_imgs, labels = album_imgs.cuda(), labels.cuda()
                outputs = model(album_imgs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad() # Xóa gradient cũ
                loss.backward() # Tính gradient
                optimizer.step() # Cập nhật trọng số
                
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=f"{loss.item():.4f}")
        
        model.eval()
        val_loss = 0.0
        val_mAP = 0.0

        with torch.no_grad():
            for album_imgs, labels in val_loader:
                album_imgs, labels = album_imgs.cuda(), labels.cuda()
                outputs = model(album_imgs)
                val_loss += criterion(outputs, labels).item()
                val_mAP += compute_mAP(outputs, labels)
        
        val_loss /= len(val_loader)
        val_mAP /= len(val_loader)
        print(f"✅ Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_mAP:.4f}\n")
        
        if val_mAP > best_val_map:
            best_val_map = val_mAP
            torch.save(model.state_dict(), 'best_model.pth')
            print("🔥 Saved model with best mAP!")
            wait = 0
        else:
            wait += 1
            print(f"⏳ No improvement in mAP for {wait} epoch(s)")

        if wait >= patience:
            print("⛔ Early stopping due to no improvement in val_mAP.")
            break
        
        scheduler.step()
        # torch.cuda.empty_cache()
