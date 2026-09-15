"""Part E: per-image condition signal from reconstruction errors.

The analysis stage is deliberately thin: it reuses the Part-D patch shard and
the trained Conv-AE checkpoint, converts per-patch reconstruction errors into
band-normalized scores, then aggregates those scores per image. The output is a
review queue, not a final damage label: ranking.csv gives sortable numbers and
the contact sheets show the worst patch plus a pixelwise error heatmap.

Default model selection is pragmatic for development: prefer the Stage-2
checkpoint (Stage-2 target), but fall back to Stage 1 so the mini dry run can
be inspected end-to-end before the long pool training exists.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from aibip import config, imgio
from aibip.patch_filter import patch_confounder_label
from aibip.part_b import _label
from aibip.part_d import (
    _build_model,
    _get_device,
    _reconstruct_errors,
    read_index,
    select_rows,
    shard_paths,
)

RANKING_FIELDNAMES = [
    "rank",
    "image_id",
    "score_status",
    "condition_score",
    "raw_condition_score",
    "score_mean_z",
    "score_p90_z",
    "score_max_z",
    "raw_score_mean_z",
    "raw_score_max_z",
    "mean_error",
    "p90_error",
    "max_error",
    "n_patches",
    "n_scored_patches",
    "n_excluded_patches",
    "excluded_frac",
    "worst_patch_idx",
    "worst_band",
    "worst_row",
    "worst_col",
    "worst_patch_confounder",
    "model_stage",
    "analysis_stage",
    "baseline_stage",
]


# --- pure scoring helpers -----------------------------------------------------

def robust_band_stats(errors: np.ndarray, bands: np.ndarray,
                      n_bands: int = config.N_BANDS, min_count: int = 1,
                      min_scale: float = 1e-6) -> dict:
    """Median/MAD scale per depth band, used for perspective-aware z scores."""
    errors = np.asarray(errors, dtype=np.float64)
    bands = np.asarray(bands, dtype=np.int64)
    if errors.ndim != 1 or bands.ndim != 1 or errors.size != bands.size:
        raise ValueError("errors and bands must be same-length 1-D arrays")
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    stats = {}
    for band in range(n_bands):
        vals = errors[bands == band]
        if vals.size < min_count:
            continue
        median = float(np.median(vals))
        mad = float(np.median(np.abs(vals - median)))
        scale = 1.4826 * mad
        if scale <= 0.0:
            std = float(vals.std())
            scale = std if std > 0.0 else min_scale
        scale = max(scale, float(min_scale))
        stats[band] = {"median": median, "scale": scale, "n": int(vals.size)}
    if not stats:
        raise ValueError("no band statistics could be computed")
    return stats


def band_z_scores(errors: np.ndarray, bands: np.ndarray, stats: dict) -> np.ndarray:
    """Convert errors to per-band z scores; missing bands use global fallback."""
    errors = np.asarray(errors, dtype=np.float64)
    bands = np.asarray(bands, dtype=np.int64)
    if errors.ndim != 1 or bands.ndim != 1 or errors.size != bands.size:
        raise ValueError("errors and bands must be same-length 1-D arrays")
    fallback = {
        "median": float(np.median(errors)),
        "scale": max(float(errors.std()), config.E_MIN_Z_SCALE),
    }
    out = np.empty_like(errors, dtype=np.float64)
    for i, (err, band) in enumerate(zip(errors, bands)):
        stat = stats.get(int(band), fallback)
        out[i] = (float(err) - stat["median"]) / stat["scale"]
    return out


def top_fraction_mean(values: np.ndarray, frac: float = 0.10) -> float:
    """Mean of the highest fraction of values, at least one value."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("top_fraction_mean needs a non-empty 1-D array")
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")
    n = max(1, math.ceil(values.size * frac))
    return float(np.sort(values)[-n:].mean())


def patch_confounder_labels(shard_path: Path, rows: list) -> list:
    """Confounder labels for every patch row in shard order."""
    shard = np.load(shard_path, mmap_mode="r")
    labels = []
    for row in rows:
        labels.append(patch_confounder_label(
            np.ascontiguousarray(shard[row["patch_idx"]]), strict=True))
    return labels


def aggregate_image_scores(rows: list, errors: np.ndarray, z_scores: np.ndarray,
                           model_stage: int, analysis_stage: int,
                           baseline_stage: int, confounder_labels: list | None = None) -> list:
    """Aggregate patch scores to one condition-signal row per image."""
    if len(rows) != len(errors) or len(rows) != len(z_scores):
        raise ValueError("rows, errors and z_scores must have the same length")
    if confounder_labels is None:
        confounder_labels = [""] * len(rows)
    if len(confounder_labels) != len(rows):
        raise ValueError("confounder_labels must match rows")
    by_image: dict = {}
    for pos, row in enumerate(rows):
        by_image.setdefault(row["image_id"], []).append(pos)
    records = []
    for image_id in sorted(by_image):
        idx = np.asarray(by_image[image_id], dtype=np.int64)
        excluded = np.asarray([bool(confounder_labels[i]) for i in idx], dtype=bool)
        score_idx = idx[~excluded]
        min_scored = max(config.E_MIN_SCORABLE_PATCHES, math.ceil(idx.size * 0.25))
        reliable = score_idx.size >= min_scored
        if score_idx.size == 0:
            score_idx = idx
        image_errors = errors[idx]
        image_z = z_scores[idx]
        scored_errors = errors[score_idx]
        scored_z = z_scores[score_idx]
        local_worst = int(score_idx[int(np.argmax(scored_z))])
        worst = rows[local_worst]
        score_p90 = top_fraction_mean(scored_z)
        raw_p90 = top_fraction_mean(image_z)
        if reliable:
            score_status = "ok"
        elif int(excluded.sum()) > 0:
            score_status = "confounded"
        else:
            score_status = "low_support"
        record = {
            "rank": 0,
            "image_id": image_id,
            "score_status": score_status,
            "condition_score": score_p90,
            "raw_condition_score": raw_p90,
            "score_mean_z": float(scored_z.mean()),
            "score_p90_z": score_p90,
            "score_max_z": float(scored_z.max()),
            "raw_score_mean_z": float(image_z.mean()),
            "raw_score_max_z": float(image_z.max()),
            "mean_error": float(image_errors.mean()),
            "p90_error": top_fraction_mean(image_errors),
            "max_error": float(image_errors.max()),
            "n_patches": int(idx.size),
            "n_scored_patches": int(score_idx.size),
            "n_excluded_patches": int(excluded.sum()),
            "excluded_frac": float(excluded.mean()),
            "worst_patch_idx": int(worst["patch_idx"]),
            "worst_band": int(worst["band"]),
            "worst_row": int(worst["row"]),
            "worst_col": int(worst["col"]),
            "worst_patch_confounder": confounder_labels[local_worst],
            "model_stage": int(model_stage),
            "analysis_stage": int(analysis_stage),
            "baseline_stage": int(baseline_stage),
        }
        records.append(record)
    records.sort(key=lambda r: (r["condition_score"], r["score_mean_z"], r["image_id"]))
    for rank, record in enumerate(records, 1):
        record["rank"] = rank
    return records


# --- model inference ----------------------------------------------------------

def checkpoint_path(stage: int) -> Path:
    """Checkpoint path for a Part-D training stage."""
    return config.RUNS_DIR / f"stage{stage}" / "best.pt"


def resolve_model_stage(stage: int | None) -> int:
    """Return requested model stage, or prefer Stage 2 with Stage 1 fallback."""
    if stage is not None:
        if not checkpoint_path(stage).exists():
            raise FileNotFoundError(f"missing checkpoint {checkpoint_path(stage)}")
        return stage
    for candidate in (2, 1):
        if checkpoint_path(candidate).exists():
            return candidate
    raise FileNotFoundError("missing Part-D checkpoint: run part_d train first")


def load_model(stage: int):
    """Load the Part-D Conv-AE checkpoint for scoring."""
    import torch
    device = _get_device()
    model = _build_model().to(device)
    checkpoint = torch.load(checkpoint_path(stage), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, device, checkpoint


def score_stage(model, device, stage: int) -> tuple:
    """Compute reconstruction errors for every patch in a Part-D stage shard."""
    shard_path, index_path = shard_paths(stage)
    if not shard_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            f"missing stage-{stage} shard/index — run: python -m aibip.part_d extract --stage {stage}")
    rows = read_index(index_path)
    errors = _reconstruct_errors(model, shard_path, rows, device)
    bands = np.asarray([row["band"] for row in rows], dtype=np.int64)
    return rows, errors, bands


def baseline_rows(checkpoint: dict, all_rows: list) -> list:
    """Validation rows from the checkpoint split; fall back to all rows."""
    val_ids = checkpoint.get("val_ids") or []
    if not val_ids:
        return all_rows
    try:
        return select_rows(all_rows, val_ids)
    except ValueError:
        return all_rows


def baseline_stats(model, device, checkpoint: dict, stage: int) -> dict:
    """Band stats from the model stage's validation split."""
    shard_path, index_path = shard_paths(stage)
    rows = read_index(index_path)
    base_rows = baseline_rows(checkpoint, rows)
    errors = _reconstruct_errors(model, shard_path, base_rows, device)
    bands = np.asarray([row["band"] for row in base_rows], dtype=np.int64)
    try:
        return robust_band_stats(
            errors, bands, min_count=config.E_MIN_BASELINE_PER_BAND,
            min_scale=config.E_MIN_Z_SCALE)
    except ValueError:
        all_errors = _reconstruct_errors(model, shard_path, rows, device)
        all_bands = np.asarray([row["band"] for row in rows], dtype=np.int64)
        return robust_band_stats(
            all_errors, all_bands, min_count=config.E_MIN_BASELINE_PER_BAND,
            min_scale=config.E_MIN_Z_SCALE)


# --- visualization ------------------------------------------------------------

def reconstruct_patch(model, device, patch: np.ndarray) -> np.ndarray:
    """One reconstructed HWC uint8 patch."""
    import torch
    x = torch.from_numpy(np.array(patch, copy=True)).permute(2, 0, 1)
    x = x.unsqueeze(0).float().div_(255.0).to(device)
    with torch.no_grad():
        out = model(x).clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    return (out * 255.0).round().astype(np.uint8)


def error_overlay(patch: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Patch with a TURBO heatmap overlay of pixelwise reconstruction error."""
    err = ((patch.astype(np.float32) - recon.astype(np.float32)) / 255.0) ** 2
    err = err.mean(axis=2)
    lo, hi = float(np.percentile(err, 5)), float(np.percentile(err, 99))
    norm = np.clip((err - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return (patch * 0.55 + heat * 0.45).round().astype(np.uint8)


def patch_tile(shard, row: dict, recon: np.ndarray, score: float,
               confounder: str = "") -> np.ndarray:
    """Tile: original patch | reconstruction | error overlay."""
    patch = np.ascontiguousarray(shard[row["patch_idx"]])
    overlay = error_overlay(patch, recon)
    tile = np.hstack([patch, recon, overlay])
    suffix = f" {confounder}" if confounder else ""
    _label(tile, (4, 18), f"{row['image_id']} b{row['band']} z={score:.2f}{suffix}")
    return tile


def write_contact_sheet(records: list, rows_by_patch: dict, z_by_patch: dict,
                        label_by_patch: dict, model, device, shard_path: Path,
                        out_path: Path) -> None:
    """Write worst-patch review sheet for a list of ranking records."""
    if not records:
        if out_path.exists():
            out_path.unlink()
        return
    shard = np.load(shard_path, mmap_mode="r")
    tiles = []
    for record in records:
        patch_idx = int(record["worst_patch_idx"])
        row = rows_by_patch[patch_idx]
        patch = np.ascontiguousarray(shard[patch_idx])
        recon = reconstruct_patch(model, device, patch)
        tiles.append(patch_tile(shard, row, recon, z_by_patch[patch_idx],
                                label_by_patch.get(patch_idx, "")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imgio.save_image(out_path, imgio.montage(tiles, cols=3, pad=4))


def write_ranking(path: Path, records: list) -> None:
    """ranking.csv from most typical (rank 1) to most anomalous (last rank)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RANKING_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def confounder_review_records(records: list, rows_by_patch: dict,
                              z_by_patch: dict, label_by_patch: dict,
                              n: int) -> list:
    """Copies of records pointing at their highest-z excluded patch."""
    patches_by_image: dict = {}
    for patch_idx, row in rows_by_patch.items():
        if label_by_patch.get(patch_idx):
            patches_by_image.setdefault(row["image_id"], []).append(patch_idx)
    out = []
    for record in records:
        patch_ids = patches_by_image.get(record["image_id"], [])
        if not patch_ids:
            continue
        patch_idx = max(patch_ids, key=lambda idx: z_by_patch[idx])
        row = rows_by_patch[patch_idx]
        copy = dict(record)
        copy.update({
            "worst_patch_idx": int(patch_idx),
            "worst_band": int(row["band"]),
            "worst_row": int(row["row"]),
            "worst_col": int(row["col"]),
            "worst_patch_confounder": label_by_patch[patch_idx],
        })
        out.append(copy)
    out.sort(key=lambda r: (-z_by_patch[int(r["worst_patch_idx"])], r["image_id"]))
    return out[:n]


def analyze(model_stage: int | None = None, analysis_stage: int = 1,
            n: int = 12) -> list:
    """CLI body: score mini patches, write ranking and top/bottom review sheets."""
    resolved_model_stage = resolve_model_stage(model_stage)
    model, device, checkpoint = load_model(resolved_model_stage)
    rows, errors, bands = score_stage(model, device, analysis_stage)
    stats = baseline_stats(model, device, checkpoint, resolved_model_stage)
    z_scores = band_z_scores(errors, bands, stats)
    shard_path, _ = shard_paths(analysis_stage)
    confounder_labels = patch_confounder_labels(shard_path, rows)
    records = aggregate_image_scores(
        rows, errors, z_scores, resolved_model_stage, analysis_stage,
        resolved_model_stage, confounder_labels)

    ranking_path = config.OUTPUTS_DIR / "ranking.csv"
    write_ranking(ranking_path, records)

    rows_by_patch = {int(row["patch_idx"]): row for row in rows}
    z_by_patch = {int(row["patch_idx"]): float(z)
                  for row, z in zip(rows, z_scores)}
    label_by_patch = {int(row["patch_idx"]): label
                      for row, label in zip(rows, confounder_labels)}
    qc_dir = config.OUTPUTS_DIR / "qc"
    reliable_records = [r for r in records if r["score_status"] == "ok"]
    if not reliable_records:
        reliable_records = records
    bottom = reliable_records[:min(n, len(reliable_records))]
    top = list(reversed(reliable_records[-min(n, len(reliable_records)):]))
    confounded = confounder_review_records(
        records, rows_by_patch, z_by_patch, label_by_patch, min(n, len(records)))
    write_contact_sheet(bottom, rows_by_patch, z_by_patch, label_by_patch, model,
                        device, shard_path, qc_dir / "e_most_typical.jpg")
    write_contact_sheet(top, rows_by_patch, z_by_patch, label_by_patch, model,
                        device, shard_path, qc_dir / "e_most_anomalous.jpg")
    write_contact_sheet(confounded, rows_by_patch, z_by_patch, label_by_patch, model,
                        device, shard_path, qc_dir / "e_confounders.jpg")

    label = "Stage-2" if resolved_model_stage == 2 else "Stage-1 dry-run"
    print(f"analyze(E): {len(records)} images scored with {label} checkpoint "
          f"on {device.type} -> {ranking_path}")
    if resolved_model_stage != 2:
        print("  note: target is the Stage-2 checkpoint; this is a dry-run analysis.")
    print(f"  most typical:   {records[0]['image_id']} score {records[0]['condition_score']:.2f}")
    print(f"  most anomalous: {records[-1]['image_id']} score {records[-1]['condition_score']:.2f}")
    n_excluded = sum(bool(label) for label in confounder_labels)
    print(f"  excluded confounder patches: {n_excluded}/{len(confounder_labels)} "
          f"(vegetation/marking/shadow heuristics)")
    print(f"  wrote {qc_dir / 'e_most_typical.jpg'}, {qc_dir / 'e_most_anomalous.jpg'} "
          f"and {qc_dir / 'e_confounders.jpg'}")
    return records


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Part E: per-image condition signal and ranking")
    sub = parser.add_subparsers(dest="command", required=True)
    def add_analyze_args(command_parser) -> None:
        command_parser.add_argument(
            "--model-stage", type=int, choices=(1, 2), default=None,
            help="checkpoint stage; default: stage2 if present, else stage1")
        command_parser.add_argument(
            "--analysis-stage", "--stage", dest="analysis_stage", type=int,
            choices=(1, 2), default=1,
            help="patch shard to score; default: stage1 mini")
        command_parser.add_argument(
            "--n", type=int, default=12,
            help="images per top/bottom contact sheet")

    analyze_parser = sub.add_parser("analyze", help="score images + write ranking/QC sheets")
    add_analyze_args(analyze_parser)
    run_parser = sub.add_parser("run", help="alias for analyze")
    add_analyze_args(run_parser)
    qc_parser = sub.add_parser("qc", help="alias for analyze; regenerates E QC sheets")
    add_analyze_args(qc_parser)
    args = parser.parse_args(argv)
    if args.command in {"analyze", "run", "qc"}:
        analyze(model_stage=args.model_stage, analysis_stage=args.analysis_stage, n=args.n)


if __name__ == "__main__":
    main()
