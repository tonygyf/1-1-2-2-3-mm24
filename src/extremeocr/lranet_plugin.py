"""Imported only inside the separate LRANet++ / MMOCR environment."""
import torch
from mmdet.models.builder import DETECTORS
from mmocr.models.textend2end.spotters.LRANet_PP import LRANet_PP

from .adapter import SaturationAdapter


@DETECTORS.register_module()
class HighlightLRANetPP(LRANet_PP):
    def __init__(self, adapter_channels=256, adapter_levels=4,
                 adapter_bottleneck=32, **kwargs):
        super().__init__(**kwargs)
        self.highlight_adapter = SaturationAdapter(adapter_channels, adapter_levels,
                                                   adapter_bottleneck)
        self.register_buffer("rgb_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def extract_feat(self, img):
        features = super().extract_feat(img)
        rgb01 = (img * self.rgb_std + self.rgb_mean) / 255
        return self.highlight_adapter(features, rgb01.clamp(0, 1))
