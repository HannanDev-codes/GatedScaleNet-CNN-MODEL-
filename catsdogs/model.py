"""The neural networks.

Three choices:
  * "simple"       - a small CNN written from scratch (good to learn from)
  * "resnet18"      - a well-known CNN. With pretrained=True it already "knows"
                       how to see edges, textures and shapes, so it learns from
                       only a few images much better.
  * "gatedscalenet" - our own architecture: multi-scale gated blocks (Inception-
                       style branches + SKNet-style per-channel gating) with
                       proper downsampling between stages and deep supervision
                       instead of runtime early-exit. See gated_net.py for the
                       full design notes and reasoning. ~3.3M parameters,
                       trained from scratch (no pretrained weights exist for a
                       brand-new architecture).
"""
import torch
from torch import nn
from torchvision import models

from .gated_net import GatedScaleNet

ARCHS = ("simple", "resnet18", "gatedscalenet")


class ConvBlock(nn.Sequential):
    """Conv -> BatchNorm -> ReLU -> MaxPool.

    Conv      : slides small 3x3 "filters" over the image to find patterns (edges, fur, ears...)
    BatchNorm : keeps the numbers well-behaved so training is stable
    ReLU      : keeps only the positive signals ("this pattern was found")
    MaxPool   : shrinks the picture by 2x, keeping the strongest signal in each 2x2 patch
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )


class SimpleCNN(nn.Module):
    """4 ConvBlocks (they learn bigger and bigger patterns) + a small decision head."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),     # finds edges and colours
            ConvBlock(32, 64),    # finds textures, fur patterns
            ConvBlock(64, 128),   # finds parts: eyes, ears, nose
            ConvBlock(128, 256),  # finds whole-face shapes
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # average each of the 256 maps into one number
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),  # randomly switches off neurons so the model can't just memorise
            nn.Linear(256, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_model(arch: str = "simple", num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    if arch == "simple":
        return SimpleCNN(num_classes)
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    if arch == "gatedscalenet":
        if pretrained:
            raise ValueError(
                "gatedscalenet has no pretrained weights — it's a new architecture, "
                "nobody has trained it on a large dataset yet. Use pretrained=False."
            )
        return GatedScaleNet(num_classes)
    raise ValueError(f"Unknown arch '{arch}'. Choose from {ARCHS}.")


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    """The layer whose output Grad-CAM uses to draw the 'where did the model look' heatmap."""
    if arch == "simple":
        return model.features[-1][2]  # the last ReLU, just before the final MaxPool
    if arch == "resnet18":
        return model.layer4
    if arch == "gatedscalenet":
        return model.block4  # the last gated block, right before the head
    raise ValueError(f"Unknown arch '{arch}'.")
