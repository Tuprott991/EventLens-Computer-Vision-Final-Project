import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

# --- Transformer Encoder Layer with Attention Extraction ---
class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        src2 = attn_output
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

# --- Transformer Encoder Stack that returns all attentions ---
class TransformerEncoderWithAttn(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        attentions = []
        output = src
        for layer in self.layers:
            output, attn_weights = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask
            )
            attentions.append(attn_weights)
        return output, attentions

# --- EventLens: Multi-Label Album Event Classification ---
class EventLens(nn.Module):
    def __init__(
        self,
        num_labels,
        d_model=512,
        nhead=8,
        num_layers=8,
        max_images=20,
        backbone_name='vit_base_patch16_224',  # Sử dụng ViT-B/16
        pretrained_backbone=True
    ):
        super().__init__()
        # 1) Image feature extractor (ViT-B/16 backbone)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            num_classes=0,  # Bỏ classification head (MLP head)
            global_pool=''  # Tắt global pooling mặc định để lấy raw features
        )

        # ViT-B/16 có đầu ra 768 chiều
        feat_dim = self.backbone.num_features  # 768 cho ViT-B/16
        self.proj = nn.Linear(feat_dim, d_model)  # Project từ 768 xuống 512

        # 2) Learnable CLS token & positional embeddings cho album
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_images + 1, d_model))

        # 3) Transformer Encoder
        encoder_layer = TransformerEncoderLayerWithAttn(d_model, nhead)
        self.transformer_encoder = TransformerEncoderWithAttn(encoder_layer, num_layers)

        # 4) Classification head (multi-label)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(d_model, num_labels)

        # Khởi tạo embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, images, mask=None):
        b, n, c, h, w = images.shape
        imgs = images.view(b * n, c, h, w)
        feats = self.backbone(imgs)  # (b*n, 768) từ ViT
        feats = feats.view(b, n, -1)  # (b, n, 768)
        feats = self.proj(feats)  # (b, n, d_model)

        cls_tokens = self.cls_token.expand(b, -1, -1)  # (b, 1, d_model)
        x = torch.cat((cls_tokens, feats), dim=1)  # (b, n+1, d_model)
        x = x + self.pos_embed[:, : n + 1, :]  # Thêm positional embeddings
        x = x.transpose(0, 1)  # -> (seq_len, b, d_model)

        key_padding_mask = mask if mask is not None else None
        x, attentions = self.transformer_encoder(
            x,
            mask=None,
            src_key_padding_mask=key_padding_mask
        )

        album_repr = x[0]  # (b, d_model)
        album_repr = self.norm(album_repr)
        album_repr = self.dropout(album_repr)
        logits = self.classifier(album_repr)  # (b, num_labels)
        return logits, attentions

# --- Attention Visualization Utility ---
def visualize_attention(attentions, layer=-1, image_names=None):
    import matplotlib.pyplot as plt
    attn = attentions[layer]
    attn_avg = attn.mean(dim=1)
    cls_to_imgs = attn_avg[0, 0, 1:].detach().cpu().numpy()
    N = cls_to_imgs.shape[0]

    plt.figure()
    plt.bar(range(N), cls_to_imgs)
    plt.xlabel("Image Index")
    plt.ylabel("Attention Score")
    plt.title(f"Layer {layer} CLS→Image Attention")
    labels = image_names if image_names is not None else list(range(N))
    plt.xticks(range(N), labels, rotation=45)
    plt.tight_layout()
    plt.savefig('attention_visualization.png')

# --- Example Dataset (giả lập) ---
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, num_images, num_labels):
        self.num_samples = num_samples
        self.num_images = num_images
        self.num_labels = num_labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Giả lập ảnh: (num_images, 3, 224, 224)
        images = torch.randn(self.num_images, 3, 224, 224)
        # Giả lập nhãn: (num_labels,)
        labels = torch.randint(0, 2, (self.num_labels,), dtype=torch.float)
        return images, labels

# --- Training Loop with Freeze/Unfreeze ---
if __name__ == "__main__":
    # Config từ ảnh bạn cung cấp
    BATCH_SIZE = 6  # Batch size ban đầu
    NEW_BATCH_SIZE = 4  # Batch size sau khi unfreeze
    FREEZE_EPOCHS = 5
    EPOCHS = 20
    LEARNING_RATE = 3e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_LABELS = 23
    MAX_IMAGES = 20

    # Khởi tạo dataset và dataloader
    dataset = DummyDataset(num_samples=100, num_images=6, num_labels=NUM_LABELS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Khởi tạo mô hình
    model = EventLens(
        num_labels=NUM_LABELS,
        d_model=512,
        nhead=8,
        num_layers=8,
        max_images=MAX_IMAGES,
        backbone_name='vit_base_patch16_224',
        pretrained_backbone=True
    ).to(DEVICE)

    # Tiêu chí loss cho multi-label classification
    criterion = nn.BCEWithLogitsLoss()

    # --- Giai đoạn 1: Freeze backbone và huấn luyện ---
    print("Phase 1: Training with frozen backbone...")
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    model.train()
    for epoch in range(FREEZE_EPOCHS):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{FREEZE_EPOCHS}, Loss: {total_loss/len(dataloader)}")

    # Lưu model sau giai đoạn 1
    torch.save(model.state_dict(), "model_after_freeze.pth")
    print("Model saved after freeze phase.")

    # --- Giai đoạn 2: Unfreeze backbone, giảm batch size, và fine-tune ---
    print("Phase 2: Unfreezing backbone and fine-tuning...")
    dataloader = DataLoader(dataset, batch_size=NEW_BATCH_SIZE, shuffle=True)  # Giảm batch size

    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE / 2)  # Giảm learning rate khi fine-tune

    model.train()
    for epoch in range(FREEZE_EPOCHS, EPOCHS):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader)}")

    # Lưu model cuối cùng
    torch.save(model.state_dict(), "model_final.pth")
    print("Final model saved.")

    # Visualize attention từ epoch cuối
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(DEVICE)
            _, attentions = model(images)
            visualize_attention(attentions, layer=-1, image_names=[f"img{i}" for i in range(6)])
            break  # Chỉ visualize batch đầu tiên