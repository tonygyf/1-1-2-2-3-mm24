"""Hypothesis A1: saturation-conditioned residuals on FPN feature maps."""
import torch
from torch import nn
from torch.nn import functional as F


class SaturationAdapter(nn.Module):
    def __init__(self, channels=256, levels=4, bottleneck=32):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels + 2, bottleneck, 1), nn.ReLU(),
                          nn.Conv2d(bottleneck, bottleneck, 3, padding=1,
                                    groups=bottleneck), nn.ReLU(),
                          nn.Conv2d(bottleneck, channels, 1))
            for _ in range(levels)
        ])
        for branch in self.branches:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def forward(self, features, rgb01):
        if len(features) != len(self.branches):
            raise ValueError("FPN level count differs from adapter config")
        # Soft indicators allow conditioning on near saturation, including color clipping.
        any_clip = ((rgb01.amax(dim=1, keepdim=True) - 0.90) / 0.10).clamp(0, 1)
        all_clip = ((rgb01.amin(dim=1, keepdim=True) - 0.90) / 0.10).clamp(0, 1)
        maps = torch.cat([any_clip, all_clip], dim=1)
        output = []
        for feature, branch in zip(features, self.branches):
            condition = F.interpolate(maps, size=feature.shape[-2:], mode="area")
            residual = branch(torch.cat([feature, condition], dim=1))
            output.append(feature + residual)
        return tuple(output)
