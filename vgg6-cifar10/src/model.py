"""VGG6 (conv64-conv64-pool-conv128-conv128-pool-GAP-fc) with configurable activation."""
import math

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}

# Architecture variants (all end with global-average-pool + one linear layer).
#   template  : the assignment template (4 conv + 1 FC)
#   six_layer : 5 conv + 1 FC = literally 6 weight layers, one extra 256-ch conv block
#   wide      : template with 2x channels (4 conv + 1 FC)
ARCHS = {
    "template": [64, 64, "M", 128, 128, "M"],
    "six_layer": [64, 64, "M", 128, 128, "M", 256, "M"],
    "wide": [128, 128, "M", 256, 256, "M"],
}


def make_layers(cfg, activation: str = "relu", batch_norm: bool = True) -> nn.Sequential:
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unknown activation {activation!r}; choose from {list(ACTIVATIONS)}")
    layers, in_ch = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        layers.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(v))
        layers.append(ACTIVATIONS[activation]())
        in_ch = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features: nn.Module, in_features: int, num_classes: int = 10):
        super().__init__()
        self.features = features
        self.classifier = nn.Linear(in_features, num_classes)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # Global average pooling. Equivalent to adaptive_avg_pool2d(x, 1) but its
        # backward is deterministic on CUDA (helps exact reproducibility).
        x = x.mean(dim=(2, 3))
        return self.classifier(x)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_vgg6(activation: str = "relu", batch_norm: bool = True, num_classes: int = 10,
               arch: str = "template") -> VGG:
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch {arch!r}; choose from {list(ARCHS)}")
    cfg = ARCHS[arch]
    last_ch = [v for v in cfg if v != "M"][-1]
    return VGG(make_layers(cfg, activation, batch_norm), last_ch, num_classes)
