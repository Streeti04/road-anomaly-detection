"""Small patch-level heuristics shared by Parts D and E."""

import cv2
import numpy as np


def _has_elongated_component(mask: np.ndarray, min_area_frac: float = 0.012,
                             min_aspect: float = 2.5,
                             min_long_side_frac: float = 0.25) -> bool:
    """True for long, thin connected components such as road markings."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    h, w = mask.shape[:2]
    min_area = min_area_frac * h * w
    min_long_side = min_long_side_frac * min(h, w)
    for label in range(1, n_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bw = float(stats[label, cv2.CC_STAT_WIDTH])
        bh = float(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = max(bw / max(bh, 1.0), bh / max(bw, 1.0))
        if aspect >= min_aspect and max(bw, bh) >= min_long_side:
            return True
    return False


def _looks_like_shadow(v: np.ndarray, s: np.ndarray) -> bool:
    """Patch contains a strong low-saturation light/dark split."""
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    contrast = float(p90 - p10)
    dark = (v <= min(95.0, p50 - 18.0)) & (s <= 95)
    bright = (v >= p50 + 18.0) & (s <= 115)
    if contrast >= 45.0 and float(dark.mean()) >= 0.18 and float(bright.mean()) >= 0.12:
        return True
    dark_abs = (v <= 90) & (s <= 120)
    bright_abs = (v >= 112) & (s <= 135)
    if contrast >= 35.0 and float(dark_abs.mean()) >= 0.22 and float(bright_abs.mean()) >= 0.08:
        return True
    very_dark = (v <= 70) & (s <= 130)
    return contrast >= 30.0 and float(very_dark.mean()) >= 0.28 and float(p90) >= 95.0


def _looks_like_low_light_shadow(v: np.ndarray, s: np.ndarray) -> bool:
    """Strict E-only case: mostly dark, low-saturation road patch."""
    p50, p90 = np.percentile(v, [50, 90])
    low_sat = float((s <= 125).mean())
    very_dark = float(((v <= 85) & (s <= 135)).mean())
    return float(p50) <= 88.0 and float(p90) <= 132.0 and low_sat >= 0.75 and very_dark >= 0.45


def patch_confounder_label(patch: np.ndarray, strict: bool = False) -> str:
    """Return a conservative label for obvious non-damage patch confounders.

    The goal is not semantic segmentation. These color rules only catch cases
    that should not define "normal road" during training and should not drive
    the final road-condition score: vegetation at mask edges, road markings
    and strong cast shadows. `strict=True` is intended for Part E's review
    ranking only; Part D uses the default so training does not lose too many
    road-texture patches.
    """
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError("patch must be HxWx3 BGR")
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (h >= 35) & (h <= 95) & (s >= 40) & (v >= 35)
    dry_plant = (h >= 8) & (h <= 34) & (s >= 45) & (v >= 55)
    white = (s <= 85) & (v >= 145)
    strict_marking = (s <= 100) & (v >= 128)
    labels = []
    if float(green.mean()) >= 0.06 or float(dry_plant.mean()) >= 0.18:
        labels.append("vegetation")
    if float(white.mean()) >= 0.045 or _has_elongated_component(white):
        labels.append("marking")
    elif strict and (float(strict_marking.mean()) >= 0.025 or
                     _has_elongated_component(strict_marking, min_area_frac=0.004,
                                              min_aspect=2.0,
                                              min_long_side_frac=0.14)):
        labels.append("marking")
    if _looks_like_shadow(v, s) or (strict and _looks_like_low_light_shadow(v, s)):
        labels.append("shadow")
    return "+".join(labels)
