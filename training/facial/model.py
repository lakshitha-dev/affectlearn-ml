"""CNN-LSTM architecture for DAiSEE facial affect detection.

Input:  (B, T, 3, 96, 96)   - batch of T-frame clips
Output: (B, num_classes)    - class logits
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BN -> ReLU -> MaxPool (halves spatial dims)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class FrameCNN(nn.Module):
    """Per-frame spatial encoder: (B,3,96,96) -> (B,out_dim).

    backbone='scratch'  : original 4-block CNN trained from scratch.
    backbone='resnet18' : ImageNet-pretrained ResNet18 (inputs are already
                          ImageNet-normalised by preprocess.py).
    """

    def __init__(self, out_dim: int = 512, backbone: str = "scratch"):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = net.fc.in_features          # 512
            net.fc = nn.Identity()
            self.net  = net
            self.proj = nn.Identity() if out_dim == feat_dim else nn.Linear(feat_dim, out_dim)
        elif backbone == "scratch":
            self.net = nn.Sequential(
                ConvBlock(3,   32),
                ConvBlock(32,  64),
                ConvBlock(64,  128),
                ConvBlock(128, 256),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
            )
            self.proj = nn.Linear(256, out_dim)
        else:
            raise ValueError(f"unknown backbone: {backbone}")

    def forward(self, x):              # (B, 3, 96, 96)
        return self.proj(self.net(x))  # (B, out_dim)


class CNNLSTMModel(nn.Module):
    """CNN feature extractor + LSTM temporal aggregator."""

    def __init__(
        self,
        num_classes: int = 4,
        cnn_out_dim: int = 512,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
        backbone: str = "scratch",
    ):
        super().__init__()
        self.cnn     = FrameCNN(out_dim=cnn_out_dim, backbone=backbone)
        self.lstm    = nn.LSTM(cnn_out_dim, hidden_size,
                               num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden_size, num_classes)

    def forward(self, x):                          # x: (B, T, 3, 96, 96)
        B, T, C, H, W = x.shape
        feats = self.cnn(x.view(B * T, C, H, W))   # (B*T, out_dim)
        feats = feats.view(B, T, -1)               # (B, T, out_dim)
        _, (h_n, _) = self.lstm(feats)             # h_n: (num_layers, B, hidden)
        out = self.dropout(h_n[-1])                # (B, hidden)
        return self.head(out)                      # (B, num_classes)


def build_model(cfg: dict) -> CNNLSTMModel:
    """Instantiate CNNLSTMModel from the model section of config.yaml."""
    return CNNLSTMModel(
        num_classes=cfg.get("num_classes", 4),
        cnn_out_dim=cfg.get("cnn_out_dim", 512),
        hidden_size=cfg.get("hidden_size", 256),
        num_layers=cfg.get("num_layers", 1),
        dropout=cfg.get("dropout", 0.5),
        backbone=cfg.get("backbone", "scratch"),
    )
