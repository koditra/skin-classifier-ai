import torch
import torch.nn as nn
import torchvision.models as models

class SkinModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()

        self.backbone = models.efficientnet_b3(weights="DEFAULT")

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)