import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import numpy as np
import os
import copy

# Giả định rằng bạn đã có file dataset.py và model_arch.py
from dataset import AlbumEventDataset
from model_arch import EventLens

# --- Config hyperparameters ---
JSON_PATH = '/kaggle/input/thesis-cufed/CUFED/event_type.json'
IMAGE_ROOT = '/kaggle/input/thesis-cufed/CUFED/images'
NUM_LABELS = 23   
BATCH_SIZE = 6
NEW_BATCH_SIZE = 4  # Batch size sau khi unfreeze
LEARNING_RATE = 3e-5
EPOCHS = 20
FREEZE_EPOCHS = 5
MAX_IMAGES = 20
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
    pos_weight = neg_counts / (pos_counts + 1e-5)  # Avoid division by zero
    return pos_weight

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Dataset and Dataloader ---
full_dataset = AlbumEventDataset(
    json_path=JSON_PATH,
    image_root=IMAGE_ROOT,
    transform=transform,
    max_images=MAX_IMAGES
)

val_len = int(len(full_dataset) * VAL_RATIO)
train_len = len(full_dataset) - val_len
train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- Model, Optimizer, Loss ---
model = EventLens(
    num_labels=NUM_LABELS,
    max_images=MAX_IMAGES,
    backbone_name='vit_base_patch16_224',  # Sử dụng ViT-B/16
    pretrained_backbone=True
)
model = model.to(DEVICE)

# Load pretrained weights if available
if os.path.exists('checkpoints/eventlens_final.pth'):
    model.load_state_dict(torch.load('checkpoints/eventlens_final.pth'))
    print("Pretrained weights loaded.")
else:
    print("No pretrained weights found. Training from scratch.")

# Tính positive weights cho BCEWithLogitsLoss
print("Calculating positive weights for BCEWithLogitsLoss...")
train_labels = torch.tensor(train_dataset.dataset.get_labels(), dtype=torch.float32)[train_dataset.indices].to(DEVICE)
pos_weight = compute_pos_weights(train_labels)
print(f"Positive weights: {pos_weight}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

# --- Early Stopping ---
best_val_loss = float('inf')
patience = 4
counter = 0
best_model_state = None
os.makedirs("checkpoints", exist_ok=True)

# --- Freeze/Unfreeze Backbone ---
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
        if all_labels[:, i].sum() == 0:  # Bỏ qua lớp không có nhãn dương
            continue
        ap = average_precision_score(all_labels[:, i], all_logits[:, i])
        ap_per_class.append(ap)

    mean_ap = np.mean(ap_per_class) if ap_per_class else 0.0
    return total_loss / len(dataloader), mean_ap

# --- Training Loop ---
# Giai đoạn 1: Freeze backbone
print("Phase 1: Training with frozen backbone...")
freeze_backbone(model, freeze=True)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

for epoch in range(FREEZE_EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
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

    # Checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, f"checkpoints/best_model_epoch{epoch+1}_val{val_loss:.4f}.pth")
        print("✅ Improved Val Loss — model saved.")
    else:
        counter += 1
        print(f"⏳ No improvement in val loss. Patience: {counter}/{patience}")

    # Early stopping
    if counter >= patience:
        print("🛑 Early stopping triggered.")
        break

# Lưu model sau giai đoạn freeze
torch.save(model.state_dict(), "checkpoints/model_after_freeze.pth")
print("Model saved after freeze phase.")

# Giai đoạn 2: Unfreeze backbone, giảm batch size, và fine-tune
print("\nPhase 2: Unfreezing backbone and fine-tuning...")
freeze_backbone(model, freeze=False)
train_loader = DataLoader(train_dataset, batch_size=NEW_BATCH_SIZE, shuffle=True, num_workers=4)  # Giảm batch size
val_loader = DataLoader(val_dataset, batch_size=NEW_BATCH_SIZE, shuffle=False, num_workers=4)

# Cập nhật optimizer với learning rate giảm
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 2)  # Giảm learning rate khi fine-tune

for epoch in range(FREEZE_EPOCHS, EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
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

    # Checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, f"checkpoints/best_model_epoch{epoch+1}_val{val_loss:.4f}.pth")
        print("✅ Improved Val Loss — model saved.")
    else:
        counter += 1
        print(f"⏳ No improvement in val loss. Patience: {counter}/{patience}")

    # Early stopping
    if counter >= patience:
        print("🛑 Early stopping triggered.")
        break

# --- Save the final model ---
torch.save(model.state_dict(), "checkpoints/eventlens_final.pth")
print("\nTraining complete. Model saved to checkpoints/eventlens_final.pth")