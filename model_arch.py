import torch
import torch.nn as nn
import timm

class AlbumEventClassifier(nn.Module):
    def __init__(self, 
                 backbone_name='swin_tiny_patch4_window7_224',  #  SwinV2 as backbone
                 pretrained=True,
                 num_classes=23,
                 aggregator='transformer',
                 max_images=32):
        super().__init__()

        self.max_images = max_images

        #  Load SwinV2 backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        #  Auto-detect `embed_dim`
        self.embed_dim = self.backbone.num_features  
        print(f"Using Swin-Tiny backbone: {backbone_name}, embed_dim: {self.embed_dim}")

        #  Positional Encoding for image sequence
        self.pos_embedding = nn.Parameter(torch.randn(1, max_images, self.embed_dim))

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
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, num_classes)
        )

        #  Multi-label activation
        self.activation = nn.Sigmoid()

    def forward(self, album_imgs):
        # album_imgs: (B, N, C, H, W)
        B, N, C, H, W = album_imgs.size()
        album_imgs = album_imgs.view(B * N, C, H, W)
        feats = self.backbone(album_imgs)  # (B*N, D)
        feats = feats.view(B, N, self.embed_dim)  # (B, N, D)

        # Prepare CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        feats = torch.cat((cls_tokens, feats), dim=1)  # (B, 1+N, D)
    
        # Add positional embedding 
        feats = feats + self.pos_embedding[:, :feats.size(1), :] # (B, 1+N, D)

        if isinstance(self.aggregator, nn.TransformerEncoder):
            agg = self.aggregator(feats)
            agg = agg[:, 0]  # Take CLS token representation
        elif isinstance(self.aggregator, nn.LSTM):
            _, (h_n, _) = self.aggregator(feats)
            agg = h_n[-1]
        else:
            agg = feats.mean(dim=1)

        out = self.classifier(agg)
        return self.activation(out)


def load_model(model_path, backbone_name='swin_tiny_patch4_window7_224', num_classes=23, aggregator='transformer', max_images=32):
    model = AlbumEventClassifier(backbone_name=backbone_name, num_classes=num_classes, aggregator=aggregator, max_images=max_images)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model

