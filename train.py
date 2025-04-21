import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import numpy as np
import huggingface_hub

from dataset import AlbumEventDataset
from model_arch import EventLens  # assuming your model is saved in model.py

import os
import copy

# --- Config hyperparameters ---
JSON_PATH = '/kaggle/input/thesis-cufed/CUFED/event_type.json'
IMAGE_ROOT = '/kaggle/input/thesis-cufed/CUFED/images'
NUM_LABELS = 23   
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
EPOCHS = 30
FREEZE_EPOCHS = 5
MAX_IMAGES = 30
VAL_RATIO = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_pos_weights(labels):
    """
    Compute positive weights for each class for BCEWithLogitsLoss
    labels: Tensor of shape (num_samples, num_classes)
    """
    labels = labels.float()
    n_samples, n_classes = labels.shape
    pos_counts = labels.sum(dim=0)
    neg_counts = n_samples - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-5)  # avoid division by zero
    return pos_weight

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize ảnh về kích thước phù hợp với ConvNeXt
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Mean và Std phù hợp với ConvNeXt
                         std=[0.5, 0.5, 0.5])
])

# --- Dataset and Dataloader ---
full_dataset = AlbumEventDataset(
    json_path=JSON_PATH,
    image_root=IMAGE_ROOT,
    transform=transform,
    max_images=20 
)

val_len = int(len(full_dataset) * VAL_RATIO)
train_len = len(full_dataset) - val_len
train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- Model, Optimizer, Loss ---
model = EventLens(num_labels=NUM_LABELS, max_images=MAX_IMAGES)
model = model.to(DEVICE)

# Load pretrained weights if available
# if os.path.exists('checkpoints/eventlens_final.pth'):
#     # model.load_state_dict(torch.load('checkpoints/best_model_epoch5_val0.5863.pth'))
#     model.load_state_dict(torch.load('checkpoints/eventlens_final.pth'))
#     print("Pretrained weights loaded.")
# else:
#     print("No pretrained weights found. Training from scratch.")

# Load pretrained weights from my huggingface account  if available

#Download weight from my huggingface
# !pip install huggingface_hub

# !huggingface-cli login --token hf

huggingface_model_download = huggingface_hub.hf_hub_download(repo_id="Vantuk/Eventlens_Photo_Album_Event_Recognition", filename="eventlens_convnext_v2_unfreeze_1.pth")

model.load_state_dict(torch.load(huggingface_model_download, map_location=DEVICE))

print("Calculating positive weights for BCEWithLogitsLoss...")
# Assuming AlbumEventDataset has a `labels` attribute or method
# Access preloaded labels directly from the dataset
train_labels = torch.tensor(train_dataset.dataset.get_labels(), dtype=torch.float32)[train_dataset.indices].to(DEVICE)
pos_weight = compute_pos_weights(train_labels)
print(f"Positive weights: {pos_weight}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

# --- Early Stopping ---
best_val_loss = float('inf')
patience = 6
counter = 0
best_model_state = None
os.makedirs("checkpoints", exist_ok=True)

# --- Freeze Backbone for first few epochs ---
def freeze_backbone(model, freeze=True):
    for param in model.backbone.parameters():
        param.requires_grad = not freeze

# --- Evaluation: Calculate Loss & mAP ---
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_logits = []
    total_loss = 0

    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits, _ = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        all_logits.append(logits.sigmoid().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)

    ap_per_class = []
    for i in range(all_labels.shape[1]):
        try:
            ap = average_precision_score(all_labels[:, i], all_logits[:, i])
        except:
            ap = 0.0
        ap_per_class.append(ap)

    mean_ap = np.mean(ap_per_class)
    return total_loss / len(dataloader), mean_ap

# freeze_backbone(model, freeze=True) # Freeze backbone for first few epochs

# --- Training Loop ---
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # if epoch < FREEZE_EPOCHS:
    #     freeze_backbone(model, freeze=True)
    # else:
    #     freeze_backbone(model, freeze=False)

    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    val_loss, val_map = evaluate(model, val_loader)
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.4f}")

    # --- Checkpointing ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, f"checkpoints/best_model_epoch{epoch+1}_val{val_loss:.4f}.pth")
        print("✅ Improved Val Loss — model saved.")
    else:
        counter += 1
        print(f"⏳ No improvement in val loss. Patience: {counter}/{patience}")

    # --- Early stopping ---
    if counter >= patience:
        print("🛑 Early stopping triggered.")
        break

# --- Save the model ---
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/eventlens_final.pth")
print("\nTraining complete. Model saved to checkpoints/eventlens_final.pth")