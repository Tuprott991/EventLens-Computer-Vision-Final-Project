import torch
import torch.nn as nn
import timm
from transformers import SwinModel, AutoModel, AutoConfig, SwinConfig

class AlbumEventClassifier(nn.Module):
    def __init__(self, 
                 backbone_name='microsoft/swinv2-base-patch4-window8-256',  #  Swin as backbone
                 pretrained=True,
                 num_classes=23,
                 aggregator='transformer',
                 max_images=32):
        super().__init__()

        self.max_images = max_images

        swin_config = SwinConfig.from_pretrained(backbone_name) if pretrained else SwinConfig()
        self.backbone = SwinModel.from_pretrained(backbone_name, config=swin_config) if pretrained else SwinModel(swin_config)

        #  Auto-detect `embed_dim`
        self.embed_dim = self.backbone.num_features  
        print(f"Using Swin backbone: {backbone_name}, embed_dim: {self.embed_dim}")

        #  Positional Encoding for image sequence
        # Update positional embedding initialization
        self.pos_embedding = nn.Parameter(torch.randn(1, max_images + 1, self.embed_dim))

        # Add learnable CLS token, CLS = 'classification'
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))  # (1, 1, D) 

        # Aggregator module
        if aggregator == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,  # ✅ Match SwinV2 embed_dim
                nhead=8 if self.embed_dim % 8 == 0 else 16,  # Ensure divisibility
                batch_first=True
            )
            self.aggregator = nn.TransformerEncoder(encoder_layer, num_layers=2)
        elif aggregator == 'lstm':
            self.aggregator = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True, bidirectional=False)
        elif aggregator == 'mean':
            self.aggregator = None
        else:
            raise ValueError("Invalid aggregator")

        # ✅ Classifier with regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.embed_dim, num_classes)
        )

        # Importance weight head
        self.importance_head = nn.Linear(self.embed_dim, 1)  # Output a single weight per image

        #  Multi-label activation
        # self.activation = nn.Sigmoid()

    def forward(self, album_imgs):
        # album_imgs: (B, N, C, H, W)
        B, N, C, H, W = album_imgs.size()
        album_imgs = album_imgs.view(B * N, C, H, W)

        swin_output = self.backbone(pixel_values=album_imgs)
        feats = swin_output.last_hidden_state  # (B*N, num_patches, D)

        # Aggregate over patches (mean pooling)
        feats = feats.mean(dim=1)  # (B*N, D)

        # Reshape to (B, N, D)
        feats = feats.view(B, N, self.embed_dim) # (B, N, D)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        feats = torch.cat((cls_tokens, feats), dim=1)  # (B, N+1, D)

        # Calculate importance weights
        importance_weights = self.importance_head(feats[:, 1:, :]).squeeze(-1)  # (B, N)
    
        if isinstance(self.aggregator, nn.TransformerEncoder):
            agg = self.aggregator(feats)
            agg = agg[:, 0]  # Take CLS token representation
        elif isinstance(self.aggregator, nn.LSTM):
            _, (h_n, _) = self.aggregator(feats)
            agg = h_n[-1]
        else:
            agg = feats.mean(dim=1)

        out = self.classifier(agg)
        return out, importance_weights


def load_model(model_path, backbone_name='swin_tiny_patch4_window7_224', num_classes=23, aggregator='transformer', max_images=32):


    model = AlbumEventClassifier(backbone_name=backbone_name, num_classes=num_classes, aggregator=aggregator, max_images=max_images)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        alpha: trọng số cho class dương (positive class)
        gamma: hệ số điều chỉnh độ khó của mẫu
        reduction: 'mean', 'sum', hoặc 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (B, C), đầu ra sigmoid của mô hình
        targets: (B, C), nhãn nhị phân
        """
        eps = 1e-8  # để tránh log(0)
        inputs = torch.clamp(inputs, eps, 1.0 - eps)  # tránh nan

        BCE_loss = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_term = (1 - pt) ** self.gamma

        loss = self.alpha * focal_term * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss