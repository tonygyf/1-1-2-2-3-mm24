"""Hypothesis A1: saturation-conditioned residuals on FPN feature maps."""
import torch
from torch import nn
from torch.nn import functional as F


class SaturationAdapter(nn.Module):
    """Equal-capacity controls differ only in the two conditioning channels.

    ``zeros`` is the unconditional residual control. ``mean`` removes location;
    ``flip`` deliberately misaligns location without adding random-number calls.
    All branches exist for every variant, including level ablations.
    """
    MODES = ("saturation", "zeros", "mean", "flip")

    def __init__(self, channels=256, levels=4, bottleneck=32,
                 conditioning="saturation", active_levels=None):
        super().__init__()
        if conditioning not in self.MODES:
            raise ValueError("Unknown conditioning mode: " + conditioning)
        if min(channels, levels, bottleneck) < 1:
            raise ValueError("Adapter dimensions must be positive")
        self.conditioning = conditioning
        self.active_levels = tuple(range(levels) if active_levels is None else active_levels)
        if not self.active_levels or len(set(self.active_levels)) != len(self.active_levels):
            raise ValueError("Select distinct FPN levels")
        if any(i < 0 or i >= levels for i in self.active_levels):
            raise ValueError("FPN level is outside adapter configuration")
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels + 2, bottleneck, 1), nn.ReLU(),
                          nn.Conv2d(bottleneck, bottleneck, 3, padding=1,
                                    groups=bottleneck), nn.ReLU(),
                          nn.Conv2d(bottleneck, channels, 1))
            for _ in range(levels)
        ])
        self.reset_residual()

    def reset_residual(self):
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
        if self.conditioning == "zeros":
            maps = torch.zeros_like(maps)
        elif self.conditioning == "mean":
            maps = maps.mean(dim=(2, 3), keepdim=True).expand_as(maps)
        elif self.conditioning == "flip":
            maps = maps.flip(dims=(2, 3))
        output = []
        for level, (feature, branch) in enumerate(zip(features, self.branches)):
            if level not in self.active_levels:
                output.append(feature)
                continue
            condition = F.interpolate(maps, size=feature.shape[-2:], mode="area")
            residual = branch(torch.cat([feature, condition], dim=1))
            output.append(feature + residual)
        return tuple(output)
