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


def _label(image: np.ndarray, org: tuple, text: str) -> None:
    """White text with a 1px-offset black shadow so it reads on any background."""
    cv2.putText(image, text, (org[0] + 1, org[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def preview(data_dir, n: int = 12, seed: int = config.SEED) -> None:
    """Write a montage of the historical preprocess() work view used for D.

    The quick check when tuning ROI/config constants: run, then view
    outputs/preview_preprocess.jpg.
    """
    paths = imgio.list_images(data_dir)
    if not paths:
        raise ValueError(f"no images found in {data_dir}")
    if len(paths) > n:
        paths = random.Random(seed).sample(paths, n)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tiles = []
    for path in paths:
        out = preprocess(imgio.load_image(path))
        h, w = out.shape[:2]
        tile = imgio.resize(out, (round(h * 1024 / w), 1024))
        _label(tile, (8, 24), Path(path).stem)
        tiles.append(tile)
    out_path = config.OUTPUTS_DIR / "preview_preprocess.jpg"
    imgio.save_image(out_path, imgio.montage(tiles, cols=3, pad=6))
    print(f"wrote {out_path} ({len(tiles)} tiles, preprocess output {h}x{w})")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Part B: preprocessing flags and preview")
    sub = parser.add_subparsers(dest="command", required=True)
    prev_parser = sub.add_parser("preview", help="montage of preprocess() outputs")
    prev_parser.add_argument("data_dir", help="directory with raw street JPEGs")
    prev_parser.add_argument("--n", type=int, default=12, help="number of images")
    prev_parser.add_argument("--seed", type=int, default=config.SEED, help="sampling seed")
    args = parser.parse_args(argv)
    if args.command == "preview":
        preview(args.data_dir, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
