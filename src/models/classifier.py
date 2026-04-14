"""
Aggression Classifier Module

Binary classifier to determine if a detected dog is exhibiting aggressive behavior.
Uses transfer learning from pretrained ImageNet models.

Supported backbones: resnet18, resnet34, mobilenet_v3_small
"""

import torch
import torch.nn as nn
from torchvision import models


class AggressionClassifier(nn.Module):
    """
    Binary classifier for dog aggression detection.
    Takes a cropped dog image and predicts: aggressive or non-aggressive.
    """

    def __init__(self, backbone="resnet18", num_classes=2,
                 pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )

        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )

        elif backbone == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.backbone(x)

    def predict(self, x):
        """Run inference and return class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs

    def predict_single(self, image_tensor):
        """Predict on a single image tensor [C, H, W] -> (class_id, confidence)."""
        self.eval()
        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            logits = self.forward(image_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, cls_id = torch.max(probs, dim=1)
        return int(cls_id[0]), float(conf[0])

    def freeze_backbone(self):
        """Freeze all layers except the final classifier head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
        print("[classifier] Backbone frozen — only head is trainable")

    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[classifier] All layers unfrozen")

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
        }, path)
        print(f"[classifier] Saved to {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            backbone=checkpoint["backbone"],
            num_classes=checkpoint["num_classes"],
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print(f"[classifier] Loaded from {path}")
        return model
