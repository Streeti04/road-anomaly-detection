"""Part C: default entry points use the accepted Test road segmentation.
The earlier geometric helpers below are retained for compatibility.
"""

import argparse
import csv
import random
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

from aibip import config, imgio
from aibip.part_b import preprocess, write_flags
from aibip.stats import outlier_flag_rows, robust_z

# timm re-registers its TinyViT variants on every mobile_sam import; harmless
# noise, silenced narrowly so real warnings stay visible.
warnings.filterwarnings("ignore", message=".*Overwriting.*")

STATS_FIELDNAMES = ["image_id", "area_frac", "strip_cov", "center_cov",
                    "side_cov", "centroid_row_frac", "sam_score", "n_components"]

ROOT = Path(__file__).resolve().parents[1]
_PREDICTORS = {}  # lazy predictor singletons, keyed by backend


def _torch_device():
    """Best available torch device for MobileSAM: MPS -> CUDA -> CPU."""
    import torch
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync_device(device) -> None:
    """Synchronize async accelerators for honest timings; CPU is a no-op."""
    if str(device) == "mps":
        import torch
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def _get_predictor(backend: str | None = None):
    """Build a SAM-family predictor once and cache it module-level.

    torch/model packages are imported HERE, not at module level, so importing
    aibip.part_c (pytest gate) never loads torch or the weights.
    """
    backend = backend or config.C_BACKEND
    if backend not in {"mobile_sam", "sam"}:
        raise ValueError(f"unknown Part-C backend {backend!r}")
    if backend not in _PREDICTORS:
        device = _torch_device()
        if backend == "mobile_sam":
            from mobile_sam import SamPredictor, sam_model_registry
            sam = sam_model_registry["vit_t"](
                str(ROOT / config.MODELS_DIR / "mobile_sam_vit_t.pt"))
        else:
            from segment_anything import SamPredictor, sam_model_registry
            checkpoint = ROOT / config.MODELS_DIR / "sam_vit_b_01ec64.pth"
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"missing full-SAM checkpoint {checkpoint}. Download sam_vit_b_01ec64.pth "
                    "from Meta's Segment Anything model zoo into models/ to test --backend sam.")
            sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        _PREDICTORS[backend] = SamPredictor(sam.to(device).eval())
    return _PREDICTORS[backend]


def _points_from_fracs(point_fracs: list, label_values: list, h: int, w: int) -> tuple:
    """Fractional (x, y) points -> SAM's absolute float32 point arrays."""
    points = [[x * w, y * h] for x, y in point_fracs]
    return np.asarray(points, dtype=np.float32), np.asarray(label_values, dtype=np.int32)


def _box_from_fracs(box_frac: tuple, h: int, w: int) -> np.ndarray:
    """Fractional (x0, y0, x1, y1) box -> SAM absolute float32 box."""
    x0, y0, x1, y1 = box_frac
    return np.asarray([x0 * w, y0 * h, x1 * w, y1 * h], dtype=np.float32)


def _prompts(h: int, w: int, profile: str | None = None) -> tuple:
    """Fixed point-prompt set for the road: (points float32 (7,2), labels int32 (7,)).

    Fixed-camera prior (A.3): every frame shares the same framing, so ONE
    prompt set serves the whole pool with a single predict() call.

    Positive points on the image midline (x = 0.5w), y = 0.50h / 0.65h / 0.80h:
    near/mid/far road samples. The lowest point (0.80h = 0.72 raw H) doubles as
    the bottom-center ANCHOR of select_candidate() and sits below every
    measured hood crest (>= 0.73 raw H).
    The 0.55/0.70/0.85 ladder was tested and 0.85 rejected: 0.85
    predates the relaxed 0.80 ROI bottom edge and would land on the
    hood/board slivers that edge intentionally keeps in WE7EJ-style frames.
    The default "corridor" profile adds lower side negatives to discourage SAM
    from merging road with grass shoulders in rural scenes like WE1MY663.
    """
    profile = profile or config.C_PROMPT_PROFILE
    point_fracs = [
        (0.50, 0.50), (0.50, 0.65), (0.50, 0.80),
        (0.50, 0.02), (0.03, 0.03), (0.97, 0.03),
        (0.50, 0.97),
    ]
    labels = [1, 1, 1, 0, 0, 0, 0]
    if profile == "corridor":
        point_fracs.extend([
            (0.16, 0.62), (0.84, 0.62),
            (0.12, 0.80), (0.88, 0.80),
        ])
        labels.extend([0, 0, 0, 0])
    elif profile != "basic":
        raise ValueError(f"unknown prompt profile {profile!r}")
    return _points_from_fracs(point_fracs, labels, h, w)


def _prompt_variants(h: int, w: int) -> list:
    """Small bank of point-only road prompts.

    Keep this deliberately simple: no SAM boxes and no side-negative points
    near the road. Boxes over-steered WE1MY5SL into sidewalk/hedge; lower
    side-negatives made some roads too narrow. We only try centerline variants
    for straight, left-bending and right-bending roads.
    """
    base_neg = [(0.50, 0.02), (0.03, 0.03), (0.97, 0.03), (0.50, 0.97)]
    profiles = [
        [(0.50, 0.45), (0.50, 0.62), (0.50, 0.80)],
        [(0.46, 0.45), (0.42, 0.62), (0.36, 0.80)],
        [(0.54, 0.45), (0.58, 0.62), (0.64, 0.80)],
    ]
    variants = []
    for points in profiles:
        labels = [1] * len(points) + [0] * len(base_neg)
        prompt_points, prompt_labels = _points_from_fracs(points + base_neg, labels, h, w)
        variants.append({"points": prompt_points, "labels": prompt_labels, "box": None})
    return variants


def _select_from_preprocessed(proc: np.ndarray, backend: str | None = None) -> tuple:
    """SAM predict on a preprocessed BGR frame; returns (candidate, score)."""
    predictor = _get_predictor(backend)
    predictor.set_image(proc[:, :, ::-1])
    masks_all, scores_all = [], []
    for prompt in _prompt_variants(*proc.shape[:2]):
        masks, scores, _ = predictor.predict(
            point_coords=prompt["points"], point_labels=prompt["labels"],
            box=prompt["box"], multimask_output=True)
        masks_all.extend(masks)
        scores_all.extend(scores)
    masks = np.asarray(masks_all, dtype=bool)
    scores = np.asarray(scores_all, dtype=np.float32)
    index = select_candidate_with_image(masks, scores, proc)
    return masks[index], float(scores[index])


def _select(image: np.ndarray, backend: str | None = None) -> tuple:
    """preprocess -> SAM predict -> candidate selection; returns (candidate, score).

    SAM expects RGB while the pipeline is BGR throughout, so the reversed view
    proc[:, :, ::-1] is fed to set_image().
    """
    return _select_from_preprocessed(preprocess(image), backend=backend)


def _legacy_segment_road(image: np.ndarray, backend: str | None = None) -> np.ndarray:
    """Road mask (bool, work resolution 614x2048) for one raw BGR frame.

    Promptable segmentation with a fixed-camera prior (A.3a): preprocess()
    (Part B) fixes the geometry, ONE 7-point prompt set (see _prompts) feeds a
    single multimask predict, and the winning candidate is chosen by position
    (bottom-center anchor + bottom-strip coverage, see select_candidate) —
    NOT by SAM's own score (A.3b). Clean-up is geometric only (largest
    component + open). Expected failures (gravel shoulders, vegetation
    overhang, hood slivers, intersections) are flagged via mask stats
    outliers -> flags.csv, never auto-corrected ("flag, don't fix").
    """
    proc = preprocess(image)
    candidate, _ = _select_from_preprocessed(proc, backend=backend)
    return simple_road_mask(candidate)


def select_candidate(masks: np.ndarray, scores: np.ndarray, h: int, w: int) -> tuple:
    """Pick one of the SAM candidates by center-road geometry (A.3b).

    Lexicographic max over the key:
    1. center_hits: all three positive centerline anchors should be inside.
    2. center_cov - side_cov: the road should occupy the center corridor but
       not spill into the lower side shoulders, which catches rural grass
       merges such as WE1MY663.
    3. sam_score: pure tie-break, deterministic by candidate index.
    Pure function (no model access) — unit-testable.
    """
    center_points = [(int(y * h), int(0.50 * w)) for y in (0.50, 0.65, 0.80)]
    y0, y1 = int(0.55 * h), int(0.86 * h)
    cx0, cx1 = int(0.42 * w), int(0.58 * w)
    lx1, rx0 = int(0.28 * w), int(0.72 * w)
    keys = []
    for mask, score in zip(masks, scores):
        center_hits = sum(bool(mask[p]) for p in center_points)
        center_cov = float(mask[y0:y1, cx0:cx1].mean())
        left_cov = float(mask[y0:y1, :lx1].mean())
        right_cov = float(mask[y0:y1, rx0:].mean())
        side_cov = max(left_cov, right_cov)
        area_frac = float(mask.mean())
        # Heavy side coverage is the pattern that put WE1MY663 onto grass.
        geometry = center_cov - 0.75 * side_cov - 0.20 * max(0.0, area_frac - 0.45)
        keys.append((center_hits, geometry, center_cov, float(score)))
    best = max(range(len(keys)), key=keys.__getitem__)
    return masks[best], best


def select_candidate_with_image(masks: np.ndarray, scores: np.ndarray,
                                proc: np.ndarray) -> int:
    """Choose a SAM road candidate with a soft, simple image prior.

    There are no hard width/shape gates here. A candidate wins when it covers
    the expected central road area and is mostly grey/asphalt-like, while green
    vegetation is only a penalty. This keeps Part C debuggable and avoids the
    previous over-restricted behavior that returned narrow slivers.
    """
    h, w = proc.shape[:2]
    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (hue >= 35) & (hue <= 95) & (sat >= 40) & (val >= 35)
    dry_plant = (hue >= 8) & (hue <= 34) & (sat >= 45) & (val >= 55)
    asphalt_like = (sat <= 110) & (val >= 30) & (val <= 230)

    y0, y1 = int(0.42 * h), int(0.90 * h)
    center_x0, center_x1 = int(0.34 * w), int(0.66 * w)
    near_y0, near_y1 = int(0.68 * h), int(0.90 * h)
    near_x0, near_x1 = int(0.24 * w), int(0.76 * w)
    keys = []
    for mask, score in zip(masks, scores):
        area = int(mask.sum())
        if area == 0:
            keys.append((-1.0, float(score)))
            continue
        center_window = float(mask[y0:y1, center_x0:center_x1].mean())
        near_window = float(mask[near_y0:near_y1, near_x0:near_x1].mean())
        mask_green = float(green[mask].mean())
        mask_dry = float(dry_plant[mask].mean())
        mask_asphalt = float(asphalt_like[mask].mean())
        area_frac = float(mask.mean())
        score_key = (
            1.8 * center_window +
            1.2 * near_window +
            0.8 * mask_asphalt -
            1.0 * mask_green -
            0.5 * mask_dry -
            0.25 * max(0.0, area_frac - 0.55) +
            0.15 * float(score)
        )
        keys.append((score_key, float(score)))
    return max(range(len(keys)), key=keys.__getitem__)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Largest 8-connected component, morphological open, then hole-fill.

    Drops satellite blobs/specks as before, and now also closes small
    enclosed background gaps (paint markings, sun-dapples, small occluders)
    so a single mis-colored pixel cluster can't punch a "Fleck" through an
    otherwise solid road patch. An empty input stays empty.
    """
    uint = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        uint, connectivity=8)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        uint = (labels == largest).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(uint, cv2.MORPH_OPEN, kernel).astype(bool)
    if not opened.any():
        return opened
    return _fill_holes(opened)


def simple_road_mask(candidate: np.ndarray) -> np.ndarray:
    """Minimal Part-C postprocess: largest component, close gaps, fill holes."""
    cleaned = clean_mask(candidate)
    if not cleaned.any():
        return cleaned
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return clean_mask(closed.astype(bool))


def road_corridor(mask: np.ndarray) -> np.ndarray:
    """Keep a plausible center-road corridor inside a SAM mask.
    ...
    NEW: stops at config.C_ROAD_MAX_HEIGHT_FRAC * h. Rows above that are
    never written to `out`, regardless of what the raw SAM candidate does up
    there — at the vanishing point SAM sometimes fuses road + tree-canopy
    shadow into one blob, and since it's contiguous with the road, the
    largest-component filter in clean_mask() can't tell it apart. A fixed
    row cutoff (justified by the fixed-camera prior, A.3) is the reliable
    way to stop it.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape[:2]
    y_min = int(config.C_ROAD_MAX_HEIGHT_FRAC * h)
    out = np.zeros_like(mask, dtype=bool)
    center = w // 2
    for y in range(h - 1, y_min - 1, -1):          # war: range(h - 1, -1, -1)
        xs = np.flatnonzero(mask[y])
        if xs.size == 0:
            continue
        if not mask[y, center]:
            center = int(xs[np.argmin(np.abs(xs - center))])
        t = y / max(h - 1, 1)
        half_frac = (config.C_ROAD_CORE_HALF_TOP_FRAC * (1.0 - t) +
                     config.C_ROAD_CORE_HALF_BOTTOM_FRAC * t)
        half = max(config.PATCH // 2, int(round(half_frac * w)))
        x0, x1 = max(0, center - half), min(w, center + half + 1)
        row = mask[y, x0:x1]
        out[y, x0:x1] = row
        kept = np.flatnonzero(row)
        if kept.size:
            center = x0 + int(np.median(kept))
    return clean_mask(out)

def _asphalt_like(proc: np.ndarray) -> np.ndarray:
    """Loose asphalt prior: grey pixels in an asphalt-typical sat/val band,
    not green/dry vegetation, not bright markings, and not pale sidewalk/
    paver material.

    Low-saturation asphalt exists in shadows, so low sat is allowed only when
    the pixel is not bright. Bright, near-achromatic material is much more
    often sidewalk/pavers/paint and stays excluded.
    """
    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = (hue >= 35) & (hue <= 95) & (sat >= 40) & (val >= 35)
    dry_plant = (hue >= 8) & (hue <= 34) & (sat >= 50) & (val >= 65)
    marking = (sat <= 55) & (val >= 185)
    neutral_dark = (sat < config.C_ASPHALT_SAT_MIN) & (val <= config.C_ASPHALT_LOW_SAT_VAL_MAX)
    textured_grey = (sat >= config.C_ASPHALT_SAT_MIN) & (sat <= 115)
    return (neutral_dark | textured_grey) & \
           (val >= 25) & (val <= config.C_ASPHALT_VAL_MAX) & \
           ~green & ~dry_plant & ~marking

def _corridor_envelope(seed: np.ndarray) -> np.ndarray:
    """Perspective corridor around the current road centerline, independent of SAM width.

    Same y_min cutoff as road_corridor() — otherwise expand_road_surface()
    would keep painting an envelope above the horizon using a stale center
    from the last real row, and pull tree/hedge pixels back in via its own
    color search even after road_corridor() correctly dropped them.
    """
    h, w = seed.shape[:2]
    y_min = int(config.C_ROAD_MAX_HEIGHT_FRAC * h)
    envelope = np.zeros_like(seed, dtype=bool)
    center = w // 2
    for y in range(h - 1, y_min - 1, -1):          # war: range(h - 1, -1, -1)
        xs = np.flatnonzero(seed[y])
        if xs.size:
            center = int(np.median(xs))
        t = y / max(h - 1, 1)
        half_frac = (config.C_EXPAND_HALF_TOP_FRAC * (1.0 - t) +
                     config.C_EXPAND_HALF_BOTTOM_FRAC * t)
        half = max(config.PATCH, int(round(half_frac * w)))
        x0, x1 = max(0, center - half), min(w, center + half + 1)
        envelope[y, x0:x1] = True
    return envelope

def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill background holes fully enclosed by foreground (standard border
    flood-fill trick): pad with a guaranteed-background ring, flood-fill
    from the corner, anything NOT reached was an enclosed hole.
    """
    h, w = binary.shape[:2]
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = binary.astype(np.uint8)
    flood = padded.copy()
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)
    outside_bg = flood[1:-1, 1:-1] == 1
    return binary | ~outside_bg

def expand_road_surface(proc: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Expand a conservative SAM seed to adjacent asphalt-like road surface.

    MobileSAM often returns only a central road strip. This adds neighboring
    grey, non-vegetation pixels inside a perspective corridor, then keeps only
    components connected to the original seed. The seed itself is always kept.
    """
    seed = np.asarray(seed, dtype=bool)
    if seed.sum() == 0:
        return seed
    envelope = _corridor_envelope(seed)
    candidates = (_asphalt_like(proc) & envelope) | seed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    candidates = cv2.morphologyEx(candidates.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    n_labels, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(seed, dtype=bool)
    seed_labels = set(np.unique(labels[seed]).tolist())
    seed_labels.discard(0)
    for label in seed_labels:
        if label < n_labels:
            keep |= labels == label
    return clean_mask(keep | seed)


def allowed_region(mask: np.ndarray) -> np.ndarray:
    """Mask eroded by PATCH//2 (129x129 ellipse) — Part D's legal patch-center zone.

    Any 128-px patch centered on a True pixel lies fully inside the road mask,
    so edge distractors (vegetation, curbs, markings at the road border) can
    never enter a training patch (A.1b). Accepted trade-off: far-road rows
    narrower than PATCH px vanish from the allowed region entirely.
    """
    size = 2 * (config.PATCH // 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def mask_stats(mask: np.ndarray, score: float) -> dict:
    """Per-image mask statistics for mask_stats.csv and outlier flagging.

    `mask` is the SELECTED candidate BEFORE cleaning — n_components documents
    how fragmented the raw SAM proposal was, the coverage metrics how much
    road it found. Pure function — unit-testable.
    """
    h = mask.shape[0]
    n_components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
    rows = np.nonzero(mask)[0]
    return {
        "area_frac": float(mask.mean()),
        "strip_cov": float(mask[int(0.70 * h):int(0.85 * h)].mean()),
        "center_cov": float(mask[int(0.55 * h):int(0.86 * h),
                                 int(0.42 * mask.shape[1]):int(0.58 * mask.shape[1])].mean()),
        "side_cov": max(
            float(mask[int(0.55 * h):int(0.86 * h), :int(0.28 * mask.shape[1])].mean()),
            float(mask[int(0.55 * h):int(0.86 * h), int(0.72 * mask.shape[1]):].mean()),
        ),
        "centroid_row_frac": float(rows.mean()) / h if rows.size else 0.0,
        "sam_score": float(score),
        "n_components": int(n_components),
    }


def flag_stats_outliers(records: list, z_thresh: float = 3.0) -> list:
    """One flags.csv row per record/metric whose robust z exceeds z_thresh.

    Same shape as part_b.flag_outliers (aibip.stats.outlier_flag_rows):
    median/MAD z on area_frac and strip_cov; a constant metric (MAD == 0)
    yields no flags. Stable order: input record order, area_frac before
    strip_cov.
    """
    return outlier_flag_rows(records, ("area_frac", "strip_cov", "side_cov"), stage="C",
                             reason="mask_stats_outlier", z_thresh=z_thresh)


def _stats_complete(row: dict | None) -> bool:
    """True if a loaded mask_stats.csv row has all current fields."""
    return row is not None and all(field in row and row[field] != "" for field in STATS_FIELDNAMES)


def _legacy_run(data_dir, backend: str | None = None, force: bool = False) -> None:
    """CLI body: batch masks + stats + flags over every image in `data_dir`.

    skip-if-exists is the only resume principle: images whose mask PNG already
    exists are skipped (their previous mask_stats.csv row is KEPT, so stats
    and flags always cover the full dataset, not just this run's delta).
    Delete masks/ to force a full pass.
    """
    paths = imgio.list_images(data_dir)
    if not paths:
        raise ValueError(f"no images found in {data_dir}")
    config.MASKS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = config.OUTPUTS_DIR / "mask_stats.csv"
    existing: dict = {}
    if stats_path.exists():
        with open(stats_path, newline="") as f:
            existing = {row["image_id"]: row for row in csv.DictReader(f)}
    new_rows: dict = {}
    written, skipped = 0, 0
    for i, path in enumerate(paths, 1):
        stem = Path(path).stem
        mask_path = config.MASKS_DIR / f"{stem}.png"
        if mask_path.exists() and not force:
            if not _stats_complete(existing.get(stem)):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"cv2 could not read mask: {mask_path}")
                old_score = float((existing.get(stem) or {}).get("sam_score", 0.0))
                new_rows[stem] = {"image_id": stem, **mask_stats(mask.astype(bool), old_score)}
            skipped += 1
            continue
        image = imgio.load_image(path)
        proc = preprocess(image)
        candidate, score = _select_from_preprocessed(proc, backend=backend)
        mask = simple_road_mask(candidate)
        imgio.save_image(mask_path, mask.astype(np.uint8) * 255)
        new_rows[stem] = {"image_id": stem, **mask_stats(mask, score)}
        written += 1
        if i % 25 == 0:
            print(f"masks: {i}/{len(paths)} images processed")
    merged = [new_rows.get(Path(p).stem) or existing.get(Path(p).stem)
              for p in paths]
    merged = [row for row in merged if row is not None]
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(merged)
    print(f"masks(C): {written} written, {skipped} skipped, {len(paths)} total")
    flag_rows = flag_stats_outliers(merged)
    write_flags(flag_rows, stage="C")
    print(f"flags(C): {len(flag_rows)} rows over {len(merged)} stats rows")


def _legacy_benchmark(data_dir, n: int = 4, backend: str | None = None) -> None:
    """CLI body: time set_image / predict separately + projections.

    Timings bracket accelerator phases with synchronization where needed —
    MPS/CUDA dispatch asynchronously, so unsynchronized perf_counter readings
    under-report. CPU synchronization is a no-op.
    """
    paths = imgio.list_images(data_dir)[:n]
    if not paths:
        raise ValueError(f"no images found in {data_dir}")
    predictor = _get_predictor(backend)
    device = _torch_device()
    warm = preprocess(imgio.load_image(paths[0]))
    predictor.set_image(warm[:, :, ::-1])  # warmup: MPS kernel compilation
    _sync_device(device)
    set_times, pred_times = [], []
    for path in paths:
        proc = preprocess(imgio.load_image(path))
        _sync_device(device); t0 = time.perf_counter()
        predictor.set_image(proc[:, :, ::-1])
        _sync_device(device); set_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for prompt in _prompt_variants(*proc.shape[:2]):
            predictor.predict(point_coords=prompt["points"], point_labels=prompt["labels"],
                              box=prompt["box"],
                              multimask_output=True)
        _sync_device(device); pred_times.append(time.perf_counter() - t0)
    t_set, t_pred = float(np.mean(set_times)), float(np.mean(pred_times))
    per_image = t_set + t_pred
    pool_n = len(imgio.list_images(config.POOL_DIR)) if config.POOL_DIR.exists() else 12432
    print(f"benchmark(C): set_image {t_set:.3f} s, predict {t_pred:.3f} s "
          f"-> {per_image:.3f} s/image (n={len(paths)}, excl. preprocess)")
    print(f"  projection: mini(100) ~{100 * per_image:.0f} s, "
          f"pool({pool_n}) ~{pool_n * per_image / 60:.0f} min "
          f"= {pool_n * per_image / 3600:.1f} h")


def _label(image: np.ndarray, org: tuple, text: str) -> None:
    """Yellow text with a thick black outline (local QC variant of part_b._label)."""
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255),
                1, cv2.LINE_AA)


def _overlay(proc: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """BGR frame with red 2px mask contour + 0.35-alpha red fill OUTSIDE the mask.

    The road itself stays natural; everything SAM did NOT claim is tinted red.
    """
    out = proc.copy()
    outside = ~mask
    red = np.array([0, 0, 255], dtype=np.float32)
    out[outside] = (out[outside] * (1 - 0.35) + red * 0.35).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    return out


def _qc_tile(image: np.ndarray, backend: str | None = None) -> np.ndarray:
    """Overlay tile for the QC sheets: preprocess + segment + tint, 1024 wide."""
    proc = preprocess(image)
    mask = segment_road(image, backend=backend)
    h, w = proc.shape[:2]
    return imgio.resize(_overlay(proc, mask), (round(h * 1024 / w), 1024))


def _outlier_ids(k: int = 4, z_thresh: float = 3.0) -> list:
    """Up to k image_ids with the largest |robust z| over area_frac/strip_cov.

    Reads outputs/mask_stats.csv; only ids with |z| > z_thresh qualify
    (mirrors the flags(C) logic). Empty list if the file is missing, empty,
    constant, or has no outliers.
    """
    stats_path = config.OUTPUTS_DIR / "mask_stats.csv"
    if not stats_path.exists():
        return []
    with open(stats_path, newline="") as f:
        records = list(csv.DictReader(f))
    ranked = []
    for metric in ("area_frac", "strip_cov"):
        z = robust_z([float(r[metric]) for r in records])
        if z is None:
            continue
        ranked.extend((abs(float(zi)), record["image_id"])
                      for record, zi in zip(records, z))
    ranked.sort(key=lambda t: -t[0])
    ids, seen = [], set()
    for zi, image_id in ranked:
        if zi <= z_thresh or image_id in seen:
            continue
        seen.add(image_id)
        ids.append(image_id)
        if len(ids) == k:
            break
    return ids


def _legacy_qc(data_dir, n: int = 12, seed: int = config.SEED, backend: str | None = None) -> None:
    """CLI body: overlay contact sheets for a seeded sample + the worst outliers.

    Sheet 1 outputs/qc/overlays.jpg: n seeded-sample frames, red boundary
    contour + red tint OUTSIDE the road mask. Sheet 2 outputs/qc/outliers.jpg:
    the same overlay for up to 4 mask-stats outliers (skipped gracefully if
    there are none).
    """
    paths = imgio.list_images(data_dir)
    if not paths:
        raise ValueError(f"no images found in {data_dir}")
    qc_dir = config.OUTPUTS_DIR / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    if len(paths) > n:
        paths = random.Random(seed).sample(paths, n)
    tiles = []
    for path in paths:
        tile = _qc_tile(imgio.load_image(path), backend=backend)
        _label(tile, (8, 24), Path(path).stem)
        tiles.append(tile)
    overlays_path = qc_dir / "overlays.jpg"
    imgio.save_image(overlays_path, imgio.montage(tiles, cols=4))
    print(f"wrote {overlays_path} ({len(tiles)} tiles)")
    outlier_ids = _outlier_ids()
    if not outlier_ids:
        print(f"skipped {qc_dir / 'outliers.jpg'} (no mask-stats outliers)")
        return
    by_stem = {Path(p).stem: p for p in imgio.list_images(data_dir)}
    tiles = []
    for stem in outlier_ids:
        if stem not in by_stem:
            continue
        tile = _qc_tile(imgio.load_image(by_stem[stem]), backend=backend)
        _label(tile, (8, 24), stem)
        tiles.append(tile)
    if tiles:
        outliers_path = qc_dir / "outliers.jpg"
        imgio.save_image(outliers_path, imgio.montage(tiles, cols=4))
        print(f"wrote {outliers_path} ({len(tiles)} tiles)")


def _legacy_single(image_path, backend: str | None = None) -> None:
    """CLI body: recompute one mask and write a dedicated overlay for debugging."""
    image_path = Path(image_path)
    image = imgio.load_image(image_path)
    proc = preprocess(image)
    candidate, score = _select_from_preprocessed(proc, backend=backend)
    mask = simple_road_mask(candidate)
    config.MASKS_DIR.mkdir(parents=True, exist_ok=True)
    qc_dir = config.OUTPUTS_DIR / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    mask_path = config.MASKS_DIR / f"{image_path.stem}.png"
    overlay_path = qc_dir / f"c_single_{image_path.stem}_{backend or config.C_BACKEND}.jpg"
    imgio.save_image(mask_path, mask.astype(np.uint8) * 255)
    tile = imgio.resize(_overlay(proc, mask), (round(proc.shape[0] * 1024 / proc.shape[1]), 1024))
    _label(tile, (8, 24), f"{image_path.stem} score={score:.3f}")
    imgio.save_image(overlay_path, tile)
    stats = mask_stats(mask, score)
    print(f"single(C): wrote {mask_path} and {overlay_path}")
    print("  " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in stats.items()))


# Default C entry points now use the Test pipeline and an explicit B-coordinate adapter.
# Old helpers above remain for existing unit tests and historical comparisons.
from aibip.c_transfer import segment_road, run, benchmark, qc, single


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Part C: MobileSAM road segmentation (masks, benchmark, QC)")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="batch masks + mask_stats.csv + flags")
    run_parser.add_argument("data_dir", help="directory with raw street JPEGs")
    run_parser.add_argument("--backend", choices=("mobile_sam", "sam"),
                            default=config.C_BACKEND, help="segmentation backend")
    run_parser.add_argument("--force", action="store_true",
                            help="overwrite existing masks instead of skip-if-exists")
    run_parser.add_argument('--limit', type=int, default=None, help='process only the first N images')
    bench_parser = sub.add_parser("benchmark", help="time set_image/predict (T2c)")
    bench_parser.add_argument("data_dir", help="directory with raw street JPEGs")
    bench_parser.add_argument("--n", type=int, default=4, help="number of images")
    bench_parser.add_argument("--backend", choices=("mobile_sam", "sam"),
                              default=config.C_BACKEND, help="segmentation backend")
    qc_parser = sub.add_parser("qc", help="overlay contact sheets")
    qc_parser.add_argument("data_dir", help="directory with raw street JPEGs")
    qc_parser.add_argument("--n", type=int, default=12, help="number of images")
    qc_parser.add_argument("--seed", type=int, default=config.SEED,
                           help="sampling seed")
    qc_parser.add_argument("--backend", choices=("mobile_sam", "sam"),
                           default=config.C_BACKEND, help="segmentation backend")
    single_parser = sub.add_parser("single", help="debug one image mask + overlay")
    single_parser.add_argument("image_path", help="path to one raw street JPEG")
    single_parser.add_argument("--backend", choices=("mobile_sam", "sam"),
                               default=config.C_BACKEND, help="segmentation backend")
    single_parser.add_argument('--force', action='store_true', help='back up and replace an incompatible existing mask')
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.data_dir, backend=args.backend, force=args.force, limit=args.limit)
    elif args.command == "benchmark":
        benchmark(args.data_dir, n=args.n, backend=args.backend)
    elif args.command == "qc":
        qc(args.data_dir, n=args.n, seed=args.seed, backend=args.backend)
    elif args.command == "single":
        single(args.image_path, backend=args.backend, force=args.force)


if __name__ == "__main__":
    main()
