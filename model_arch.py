import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
import numpy as np
from io import BytesIO
from PIL import Image
# --- Transformer Encoder Layer with Attention Extraction ---
class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Layer norms and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: (seq_len, batch, d_model)
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        # Residual + norm
        src2 = attn_output
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward
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
        num_layers=6, # 6 , 3
        max_images=30, # 30, 20
        backbone_name = 'convnextv2_base.fcmae_ft_in22k_in1k',
        pretrained_backbone=True
    ):
        super().__init__()
        # 1) Image feature extractor (ResNet50 backbone)\
        # Use ResNet50 as the backbone for image feature extraction

        # ResNet50 backbone
        # backbone = models.resnet50(pretrained=pretrained_backbone)
        # modules = list(backbone.children())[:-1]

        # self.backbone = nn.Sequential(*modules)
        # # project to transformer dimension
        # self.proj = nn.Linear(backbone.fc.in_features, d_model)

        # ConvNeXtV2 backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            num_classes=0,      # remove classification head
        )

        feat_dim = self.backbone.num_features
        self.proj = nn.Linear(feat_dim, d_model)
        
        # 1) Image feature extractor (ConvNeXt backbone)
        # self.backbone = timm.create_model(
        #     backbone_name,
        #     pretrained=pretrained_backbone,
        #     num_classes=0,      # remove classification head
        #     global_pool=''      # disable default pooling
        # )

        # remove the classification head
        
        # feat_dim = self.backbone.num_features
        # self.proj = nn.Linear(feat_dim, d_model)
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
        # Extract features and project
        feats = self.backbone(imgs).view(b, n, -1)
        feats = self.proj(feats)  # (b, n, d_model)

        # Prepare CLS token + positional embeddings
        cls_tokens = self.cls_token.expand(b, -1, -1)       # (b, 1, d_model)
        x = torch.cat((cls_tokens, feats), dim=1)            # (b, n+1, d_model)
        x = x + self.pos_embed[:, : n + 1, :]                # add pos embeddings
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
def visualize_attention(attentions, layer=-1, image_names=None, image_folder=None, images_per_row=5):
    """
    Create a grid image of album images sorted by CLS-token attention scores.

    attentions: list of attn_weights per layer;
                each attn_weights shape: (batch, num_heads, seq_len, seq_len)
    layer: which layer to visualize (default last)
    image_names: list of image filenames
    image_folder: path to the folder containing the images
    images_per_row: Number of images per row in the grid
    Returns:
        A combined image as a NumPy array.
    """
    from PIL import Image
    import numpy as np
    import os

    # Ensure layer is an integer
    layer = int(layer)
    # Select layer and average across heads
    attn = attentions[layer]                  # (b, heads, seq_len, seq_len)
    attn_avg = attn.mean(dim=1)               # (b, seq_len, seq_len)
    # CLS -> image tokens
    cls_to_imgs = attn_avg[0, 0, 1:].detach().cpu().numpy()
    N = cls_to_imgs.shape[0]

    # Ensure the number of attention scores matches the number of images
    if image_names is not None:
        N = min(N, len(image_names))  # Limit N to the number of available images
        cls_to_imgs = cls_to_imgs[:N]  # Ensure attention scores are limited to the number of images

    # Sort images by attention scores
    sorted_indices = cls_to_imgs.argsort()[::-1]  # Descending order
    sorted_scores = cls_to_imgs[sorted_indices]

    if image_names is not None and image_folder is not None:
        sorted_image_names = [image_names[i] for i in sorted_indices]
        sorted_image_paths = [os.path.join(image_folder, img_name) for img_name in sorted_image_names]

        # Load and resize images
        images = [Image.open(img_path).resize((256, 256)) for img_path in sorted_image_paths]

        # Calculate grid dimensions
        rows = (len(images) + images_per_row - 1) // images_per_row  # Round up to the nearest row
        grid_width = images_per_row * 256
        grid_height = rows * 256

        # Create a blank canvas for the grid
        combined_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
        for idx, img in enumerate(images):
            row = idx // images_per_row
            col = idx % images_per_row
            x_offset = col * 256
            y_offset = row * 256
            combined_image.paste(img, (x_offset, y_offset))

        # Convert to NumPy array
        combined_image_array = np.array(combined_image)
        return combined_image_array
    else:
        raise ValueError("Image names or folder not provided. Cannot create combined image.")

# --- Example Usage ---
if __name__ == "__main__":
    # dummy batch of 2 albums, each with 6 images (3×224×224)
    dummy_imgs = torch.randn(2, 6, 3, 224, 224)
    model = EventLens(num_labels=5, max_images=10)

    # Forward pass
    logits, attentions = model(dummy_imgs)
    preds = torch.sigmoid(logits)
    print("Predictions shape:", preds.shape)

    # Visualize attention from the last layer
    visualize_attention(attentions, layer=-1, image_names=[f"img{i}" for i in range(6)])

    # Training stub
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = nn.BCEWithLogitsLoss()
    # for epoch in range(num_epochs):
    #     for images, labels in dataloader:
    #         logits, _ = model(images)
    #         loss = criterion(logits, labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
