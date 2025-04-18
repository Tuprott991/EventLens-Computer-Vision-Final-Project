import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
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
        backbone_name='vit_base_patch16_224',
        pretrained_backbone=True
    ):
        super().__init__()
        # 1) Image feature extractor (ViT-B/16 backbone)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            num_classes=0,  # Remove the MLP head
        )

        # ViT-B/16 feature dimension is 768
        feat_dim = self.backbone.num_features  # 768 for vit_base_patch16_224
        self.proj = nn.Linear(feat_dim, d_model)  # Project to transformer dimension

        # 2) Learnable CLS token & positional embeddings for album
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_images + 1, d_model))

        # 3) Transformer Encoder
        encoder_layer = TransformerEncoderLayerWithAttn(d_model, nhead)
        self.transformer_encoder = TransformerEncoderWithAttn(encoder_layer, num_layers)

        # 4) Classification head (multi-label)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(d_model, num_labels)

        # Initialize embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, images, mask=None):
        """
        Args:
            images: Tensor (batch_size, num_images, 3, H, W)
            mask: Bool Tensor (batch_size, num_images), True for padding positions
        Returns:
            logits: Tensor (batch_size, num_labels)
            attentions: List of attention weights per layer
        """
        b, n, c, h, w = images.shape
        # Flatten batch & image dims
        imgs = images.view(b * n, c, h, w)
        # Extract features (CLS token output) and project
        feats = self.backbone(imgs)  # (b*n, 768)
        feats = feats.view(b, n, -1)  # (b, n, 768)
        feats = self.proj(feats)  # (b, n, d_model)

        # Prepare CLS token + positional embeddings
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (b, 1_calculate, d_model)
        x = torch.cat((cls_tokens, feats), dim=1)  # (b, n+1, d_model)
        x = x + self.pos_embed[:, : n + 1, :]  # Add pos embeddings
        x = x.transpose(0, 1)  # -> (seq_len, b, d_model)

        # Key padding mask: True for padding
        key_padding_mask = mask if mask is not None else None

        # Pass through transformer
        x, attentions = self.transformer_encoder(
            x,
            mask=None,
            src_key_padding_mask=key_padding_mask
        )
        # x shape: (seq_len, b, d_model)

        # Album representation = CLS token output
        album_repr = x[0]  # (b, d_model)
        album_repr = self.norm(album_repr)
        album_repr = self.dropout(album_repr)

        # Multi-label logits
        logits = self.classifier(album_repr)  # (b, num_labels)
        return logits, attentions

# --- Attention Visualization Utility ---
def visualize_attention(attentions, layer=-1, image_names=None):
    """
    Plot CLS-token attention to each image in the album.
    """
    import matplotlib.pyplot as plt
    attn = attentions[layer]  # (b, heads, seq_len, seq_len)
    attn_avg = attn.mean(dim=1)  # (b, seq_len, seq_len)
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
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Dummy batch of 2 albums, each with 6 images (3×224×224)
    dummy_imgs = torch.randn(2, 6, 3, 224, 224)
    model = EventLens(num_labels=5, max_images=10, backbone_name='vit_base_patch16_224')

    # Forward pass
    logits, attentions = model(dummy_imgs)
    preds = torch.sigmoid(logits)
    print("Predictions shape:", preds.shape)

    # Visualize attention from the last layer
    visualize_attention(attentions, layer=-1, image_names=[f"img{i}" for i in range(6)])