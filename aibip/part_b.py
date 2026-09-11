"""Part B: work view for Part D (crop, resize, CLAHE), flags and figures."""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from aibip import config, imgio
from aibip.stats import outlier_flag_rows

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - backend must be selected first

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


def figure_roi_examples(image_paths: list, out_path, n: int = 12, seed: int = config.SEED) -> None:
    """Montage (4 cols) of frames with the ROI band highlighted and cut rows labeled."""
    if len(image_paths) > n:
        image_paths = random.Random(seed).sample(image_paths, n)
    tiles = []
    for path in image_paths:
        img = imgio.resize(imgio.load_image(path), (768, 1024))
        h, w = img.shape[:2]
        y0, y1 = int(h * config.ROI_TOP_FRAC), int(h * config.ROI_BOTTOM_FRAC)
        img[:y0] = (img[:y0] * 0.35).astype(np.uint8)
        img[y1:] = (img[y1:] * 0.35).astype(np.uint8)
        cv2.line(img, (0, y0), (w - 1, y0), (0, 0, 255), 2)
        cv2.line(img, (0, y1), (w - 1, y1), (0, 0, 255), 2)
        _label(img, (10, y0 - 8), f"ROI {config.ROI_TOP_FRAC:.2f}")
        _label(img, (10, y1 - 8), f"ROI {config.ROI_BOTTOM_FRAC:.2f}")
        tiles.append(img)
    imgio.save_image(out_path, imgio.montage(tiles, cols=4))


def _road_band_crop(proc: np.ndarray) -> np.ndarray:
    """Deterministic road-band crop of a processed frame (spectra figure region)."""
    h, w = proc.shape[:2]
    return proc[int(h * 0.60):int(h * 0.95), int(w * 0.25):int(w * 0.75)]


def _log_spectrum(gray_f32: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray_f32))))


def figure_fft(image_paths: list, out_path) -> None:
    """3x3: processed frame with road-band rectangle, spectra of full frame vs road crop."""
    if len(image_paths) != 3:
        raise ValueError(f"figure_fft needs exactly 3 images, got {len(image_paths)}")
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for i, path in enumerate(image_paths):
        proc = preprocess(imgio.load_image(path))
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY).astype(np.float32)
        crop_gray = cv2.cvtColor(_road_band_crop(proc), cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = proc.shape[:2]
        frame = proc.copy()
        cv2.rectangle(frame, (int(w * 0.25), int(h * 0.60)), (int(w * 0.75), int(h * 0.95)),
                      (0, 0, 255), 3)
        axes[i, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[i, 1].imshow(_log_spectrum(gray), cmap="gray")
        axes[i, 2].imshow(_log_spectrum(crop_gray), cmap="gray")
        axes[i, 0].set_title(Path(path).stem, loc="left", fontsize=9)
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.suptitle("Part B: log magnitude spectra — full frame vs road-band crop")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def figure_filters(image_paths: list, out_path) -> None:
    """Per image, 4 panels of the SAME road-band crop: original, Gaussian, Sobel, Canny."""
    if not 1 <= len(image_paths) <= 3:
        raise ValueError(f"figure_filters needs 1 to 3 images, got {len(image_paths)}")
    fig, axes = plt.subplots(len(image_paths), 4, figsize=(16, 4.5 * len(image_paths)),
                             squeeze=False)
    for i, path in enumerate(image_paths):
        crop = _road_band_crop(preprocess(imgio.load_image(path)))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(crop, (5, 5), 1.5)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        mag = (mag / (mag.max() + 1e-9) * 255).astype(np.uint8)
        edges = cv2.Canny(gray, 50, 150)
        panels = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                  cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB), mag, edges]
        for j, panel in enumerate(panels):
            axes[i, j].imshow(panel, cmap="gray" if j > 1 else None)
            axes[i, j].set_title(["original", "Gaussian 5x5 (s=1.5)",
                                  "Sobel magnitude", "Canny 50/150"][j], fontsize=9)
        axes[i, 0].set_title(f"{Path(path).stem} | original", fontsize=9)
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.suptitle("Part B: classical filters — asphalt texture vs surrounding vegetation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def figure_clahe_ab(image_paths: list, out_path) -> None:
    """Per image, the resized ROI band without vs with CLAHE (A/B decision figure)."""
    if len(image_paths) != 3:
        raise ValueError(f"figure_clahe_ab needs exactly 3 images, got {len(image_paths)}")
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for i, path in enumerate(image_paths):
        band = resize_band(crop_roi(imgio.load_image(path)))
        axes[i, 0].imshow(cv2.cvtColor(band, cv2.COLOR_BGR2RGB))
        axes[i, 1].imshow(cv2.cvtColor(apply_clahe(band), cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title("CLAHE off" if i == 0 else None, fontsize=9)
        axes[i, 1].set_title("CLAHE on" if i == 0 else None, fontsize=9)
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.suptitle("Part B: CLAHE A/B (shadow mitigation vs noise amplification)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


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


def figures(data_dir) -> None:
    """CLI body: run flags over `data_dir` and write all Part-B figures."""
    paths = imgio.list_images(data_dir)
    if not paths:
        raise ValueError(f"no images found in {data_dir}")
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_flags(paths)
    print(f"flags(B): {len(rows)} rows over {len(paths)} images")
    jobs = [
        (figure_roi_examples, random.Random(config.SEED).sample(paths, 12), "b_roi_examples.jpg"),
        (figure_fft, sorted(random.Random(config.SEED + 1).sample(paths, 3)), "b_fft.png"),
        (figure_filters, sorted(random.Random(config.SEED + 2).sample(paths, 3)), "b_filters.png"),
        (figure_clahe_ab, sorted(random.Random(config.SEED + 3).sample(paths, 3)), "b_clahe_ab.png"),
    ]
    for builder, picked, name in jobs:
        out_path = config.FIGURES_DIR / name
        builder(picked, out_path)
        print(f"wrote {out_path}")


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
    parser = argparse.ArgumentParser(description="Part B: preprocessing flags and figures")
    sub = parser.add_subparsers(dest="command", required=True)
    fig_parser = sub.add_parser("figures", help="write flags.csv and Part-B figures")
    fig_parser.add_argument("data_dir", help="directory with raw street JPEGs")
    prev_parser = sub.add_parser("preview", help="montage of preprocess() outputs")
    prev_parser.add_argument("data_dir", help="directory with raw street JPEGs")
    prev_parser.add_argument("--n", type=int, default=12, help="number of images")
    prev_parser.add_argument("--seed", type=int, default=config.SEED, help="sampling seed")
    filter_parser = sub.add_parser("filters", help="only render filters for three Mini images; no model or dataset scan")
    filter_parser.add_argument("data_dir")
    filter_parser.add_argument("--output", type=Path, default=config.FIGURES_DIR/'b_filters.png')
    args = parser.parse_args(argv)
    if args.command == "figures":
        figures(args.data_dir)
    elif args.command == "preview":
        preview(args.data_dir, n=args.n, seed=args.seed)
    elif args.command == "filters":
        paths = imgio.list_images(args.data_dir)
        if len(paths) < 3:
            parser.error('filters requires at least three images')
        picked = sorted(random.Random(config.SEED + 2).sample(paths, 3))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure_filters(picked, args.output)
        print(f'wrote {args.output} (3 images; no segmentation or training)')


if __name__ == "__main__":
    main()
