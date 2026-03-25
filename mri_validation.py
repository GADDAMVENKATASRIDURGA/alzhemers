"""
Strict MRI input validation shared by CLI and Streamlit app.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from PIL import Image


def _open_as_rgb(image_or_path: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")
    return Image.open(image_or_path).convert("RGB")


def validate_mri_scan(image_or_path: Union[str, Image.Image]) -> Tuple[bool, str]:
    """
    Strict heuristic validation that blocks non-MRI photos.

    Returns:
        (is_valid, message)
    """
    try:
        img = _open_as_rgb(image_or_path)
        arr = np.asarray(img, dtype=np.float32)
        h, w = arr.shape[:2]

        reasons = []

        # 1) Basic geometry: MRI slices are usually square-ish and not tiny.
        aspect = min(h, w) / max(h, w)
        if min(h, w) < 128:
            reasons.append("image is too small for MRI analysis")
        if aspect < 0.75:
            reasons.append("aspect ratio is not typical for MRI slices")

        # 2) Color checks: MRI scans are grayscale/near-grayscale.
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        mad_mean = (np.mean(np.abs(r - g)) + np.mean(np.abs(r - b)) + np.mean(np.abs(g - b))) / 3.0
        rg = r - g
        yb = 0.5 * (r + g) - b
        colorfulness = float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))
        hsv_sat = np.asarray(img.convert("HSV"), dtype=np.float32)[:, :, 1] / 255.0
        mean_saturation = float(np.mean(hsv_sat))

        if mad_mean > 6.5:
            reasons.append("image is not grayscale enough for MRI")
        if colorfulness > 12.0:
            reasons.append("image is too colorful for MRI")
        if mean_saturation > 0.12:
            reasons.append("image saturation is too high for MRI")

        # 3) MRI framing checks: many MRI slices have dark background with central brain structure.
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        dark_ratio = float(np.mean(gray < 25.0))
        if dark_ratio < 0.03:
            reasons.append("background is not dark enough for typical MRI framing")

        patch_h = max(8, int(h * 0.12))
        patch_w = max(8, int(w * 0.12))
        corners = np.concatenate(
            [
                gray[:patch_h, :patch_w].ravel(),
                gray[:patch_h, -patch_w:].ravel(),
                gray[-patch_h:, :patch_w].ravel(),
                gray[-patch_h:, -patch_w:].ravel(),
            ]
        )
        corner_mean = float(np.mean(corners))
        if corner_mean > 90.0:
            reasons.append("corner regions are too bright for typical MRI background")

        y0, y1 = int(h * 0.3), int(h * 0.7)
        x0, x1 = int(w * 0.3), int(w * 0.7)
        center = gray[y0:y1, x0:x1]
        center_mean = float(np.mean(center))
        center_std = float(np.std(center))

        if center_mean < corner_mean + 8.0:
            reasons.append("central anatomy contrast does not look like MRI")
        if center_std < 12.0:
            reasons.append("central region lacks expected MRI structural variation")

        # 4) Structure check: require some edge content.
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        grad = np.zeros_like(gray)
        grad[:, 1:] += gx
        grad[1:, :] += gy
        edge_density = float(np.mean(grad > 18.0))
        if edge_density < 0.015:
            reasons.append("image lacks structural edge content expected in MRI")

        if reasons:
            return False, "Invalid input. Please provide a valid MRI scanned brain image."

        return True, "Accepted: MRI-like brain scan."

    except Exception as exc:
        return False, f"Unable to validate image: {exc}"
