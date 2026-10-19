"""Deterministic screening corruptions, not a calibrated camera simulator."""
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

KINDS = ("exposure", "glare", "bloom", "motion_blur", "rain", "snow")
VERSION = "screen-v1"


def stable_seed(seed, key):
    payload = (str(seed) + ":" + key).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def to_srgb(linear):
    x = np.clip(linear, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def corrupt(image, kind, severity, seed=42):
    """Return RGB image and metadata; coordinates and dimensions are unchanged.

    A fixed seed keeps glare location / weather geometry paired across severities.
    Severity is a generator setting, not an assertion about human readability.
    """
    if kind != "clean" and kind not in KINDS:
        raise ValueError("Unknown corruption: " + kind)
    if kind != "clean" and severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2 or 3")
    image = image.convert("RGB")
    x = np.asarray(image, dtype=np.float32) / 255
    rng = np.random.default_rng(seed)
    h, w = x.shape[:2]
    params = {}
    if kind == "clean":
        y = x
    elif kind == "exposure":
        ev = (0.75, 1.5, 2.5)[severity - 1]
        y = to_srgb(to_linear(x) * 2 ** ev)
        params = {"ev": ev}
    elif kind == "glare":
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = rng.uniform(0.15, 0.85, 2)
        radius = (0.10, 0.17, 0.24)[severity - 1]
        field = np.exp(-0.5 * (((xx / w - cx) / radius) ** 2 +
                               ((yy / h - cy) / (radius * 0.65)) ** 2))
        strength = (0.5, 1.25, 2.5)[severity - 1]
        y = to_srgb(to_linear(x) + field[..., None] * strength)
        params = {"center_xy_normalized": [float(cx), float(cy)],
                  "radius": radius, "strength": strength}
    elif kind == "bloom":
        linear = to_linear(x)
        bright = np.clip((linear.max(axis=2) - 0.65) / 0.35, 0, 1)
        halo = Image.fromarray(np.uint8(bright * 255)).filter(
            ImageFilter.GaussianBlur(min(w, h) * (0.01, 0.025, 0.05)[severity - 1]))
        y = to_srgb(linear + np.asarray(halo)[..., None] / 255 * severity * 0.7)
        params = {"source_threshold_linear": 0.65, "strength": severity * 0.7}
    elif kind == "motion_blur":
        # Horizontal line kernel with edge padding: no wrap-around artifacts.
        length = max(3, int(min(w, h) * (0.015, 0.035, 0.065)[severity - 1]) | 1)
        pad = length // 2
        padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="edge")
        cumulative = np.pad(padded.cumsum(axis=1, dtype=np.float64),
                            ((0, 0), (1, 0), (0, 0)))
        y = (cumulative[:, length:] - cumulative[:, :-length]) / length
        params = {"kernel_length": length, "angle_degrees": 0}
    else:
        layer = Image.new("L", image.size)
        draw = ImageDraw.Draw(layer)
        # Generate all particles in a stable order so levels share geometry.
        count = max(10, int(w * h / 1500))
        for _ in range(count):
            px, py = rng.uniform(0, w), rng.uniform(0, h)
            if kind == "rain":
                length = max(3, h * 0.035)
                draw.line((px, py, px - length * 0.25, py + length),
                          fill=255, width=max(1, int(w / 700)))
            else:
                radius = max(1, min(w, h) * rng.uniform(0.002, 0.009))
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=255)
        layer = layer.filter(ImageFilter.GaussianBlur(0.6))
        alpha = np.asarray(layer, dtype=np.float32)[..., None] / 255
        alpha *= (0.3, 0.55, 0.8)[severity - 1]
        y = x * (1 - alpha) + alpha
        params = {"particles": count, "opacity": (0.3, 0.55, 0.8)[severity - 1]}
    result = Image.fromarray(np.uint8(np.clip(y * 255 + 0.5, 0, 255)))
    out = np.asarray(result)
    meta = {"kind": kind, "severity": severity, "seed": seed,
            "generator": VERSION, "parameters": params,
            "all_channel_clip_fraction": float((out.min(axis=2) >= 250).mean()),
            "any_channel_clip_fraction": float((out.max(axis=2) >= 250).mean())}
    return result, meta
