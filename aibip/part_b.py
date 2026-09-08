"""Part B: work view for Part D (crop, resize, CLAHE), flags and figures."""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from aibip import config, imgio
from aibip.stats import outlier_flag_rows

FLAG_FIELDNAMES = ["image_id", "stage", "reason", "metric", "value", "z"]


def crop_roi(image: np.ndarray) -> np.ndarray:
    """Crop the fixed ROI band rows [ROI_TOP_FRAC, ROI_BOTTOM_FRAC) * h, full width.

    Fractions were measured on this campaign's framing; they are applied
    to the PASSED image height (not config.RAW_RES) so any input size works.
    """
    h = image.shape[0]
    y0, y1 = int(h * config.ROI_TOP_FRAC), int(h * config.ROI_BOTTOM_FRAC)
    if not (0 <= y0 < y1 <= h):
        raise ValueError(f"image too small for ROI crop: h={h}, band=[{y0}, {y1})")
    return image[y0:y1]


def resize_band(image: np.ndarray) -> np.ndarray:
    """Aspect-preserving resize of the ROI band to width WORK_WIDTH (INTER_AREA)."""
    band_h, band_w = image.shape[:2]
    out_w = config.WORK_WIDTH
    out_h = max(1, round(band_h * out_w / band_w))
    return imgio.resize(image, (out_h, out_w))


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """CLAHE (clip 2.0, 8x8 tiles) on the LAB L channel; in/out BGR uint8."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(image: np.ndarray) -> np.ndarray:
    """D work view: 40--80% crop, resize to WORK_WIDTH, CLAHE on L channel.

    Fixed crop instead of per-image hood detection: deterministic and cheap;
    residual intrusions are flagged via hood_strip_stats -> flags.csv
    ("flag, don't fix").
    """

    band = crop_roi(image)
    out = resize_band(band)
    return apply_clahe(out) if config.USE_CLAHE else out


def hood_strip_stats(image: np.ndarray) -> dict:
    """Mean HSV V/S over rows [ROI_BOTTOM_FRAC - 0.06, ROI_BOTTOM_FRAC) * h.

    This strip just above the ROI bottom edge is normal road in typical frames;
    hood / chevron-board intrusion makes it dark / saturated.
    """
    h = image.shape[0]
    y0 = int(h * (config.ROI_BOTTOM_FRAC - 0.06))
    y1 = int(h * config.ROI_BOTTOM_FRAC)
    if not (0 <= y0 < y1 <= h):
        raise ValueError(f"image too small for hood strip: h={h}, strip=[{y0}, {y1})")
    hsv = cv2.cvtColor(image[y0:y1], cv2.COLOR_BGR2HSV)
    return {"v": float(hsv[..., 2].mean()), "s": float(hsv[..., 1].mean())}


def flag_outliers(records: list, z_thresh: float = 3.0) -> list:
    """Flag records whose hood-strip v/s is a robust-z outlier (median/MAD).

    A constant metric (MAD == 0) yields no flags. One row per offending metric;
    stable order: input record order, metric 'v' before 's'.
    """
    return outlier_flag_rows(records, ("v", "s"), stage="B",
                             reason="hood_strip_outlier", z_thresh=z_thresh)


def write_flags(flag_rows: list, stage: str = "B") -> None:
    """Rewrite config.FLAGS_CSV: drop old rows of `stage`, keep every other stage.

    Each pipeline stage (B preprocessing, C segmentation, ...) owns its rows;
    rewriting one stage never touches the others.
    """
    config.FLAGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    if config.FLAGS_CSV.exists():
        with open(config.FLAGS_CSV, newline="") as f:
            kept = [row for row in csv.DictReader(f) if row.get("stage") != stage]
    with open(config.FLAGS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FLAG_FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(flag_rows)


def run_flags(image_paths: list) -> list:
    """Load every image, collect hood-strip stats, flag outliers, write flags.csv."""
    records = []
    for i, path in enumerate(image_paths, 1):
        stats = hood_strip_stats(imgio.load_image(path))
        records.append({"image_id": Path(path).stem, **stats})
        if i % 25 == 0:
            print(f"flags: {i}/{len(image_paths)} images processed")
    rows = flag_outliers(records)
    write_flags(rows)
    return rows


