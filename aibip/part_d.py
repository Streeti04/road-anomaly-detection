"""Part D: model helpers plus the auditable Mini-only D v2 command entry points.

New extraction/training live in d_data/d_training.
The existing AE architecture and historical E-facing helper functions remain.
Legacy workflows are retained under _legacy_* names but are not CLI defaults.
All Torch imports remain lazy; importing this module does not load a model.
"""

import argparse
import csv
import math
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

from aibip import config, imgio
from aibip.patch_filter import patch_confounder_label
from aibip.part_b import _label, preprocess
# D-only square clearance; the accepted C segmentation is unchanged.
def allowed_region(mask):
    size = 2 * (config.PATCH // 2) + 1
    return cv2.erode(mask.astype(np.uint8), np.ones((size, size), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

INDEX_FIELDNAMES = ["patch_idx", "image_id", "band", "row", "col"]


# --- pure helpers (torch-free, unit-tested) -----------------------------------

def patch_center_seed(image_id: str, base_seed: int = config.SEED) -> int:
    """Stable per-image RNG seed: base_seed XOR crc32(image_id).

    crc32 instead of Python's hash() — the latter is salted per process and
    would break cross-run determinism. The mask makes the seed platform-
    independent 32-bit.
    """
    if not image_id:
        raise ValueError("image_id must be a non-empty string")
    return (base_seed ^ zlib.crc32(image_id.encode("utf-8"))) & 0xFFFFFFFF


def band_edges(road_rows: np.ndarray, n_bands: int = config.N_BANDS) -> np.ndarray:
    """Integer boundaries e_0..e_n evenly splitting the road-row span into bands.

    Band b covers rows [e_b, e_{b+1}); the last edge is max+1, so the deepest
    road row lands in band n_bands-1. Raises if the span is degenerate —
    masks that narrow cannot yield PATCH-sized centers anyway (C.3).
    """
    road_rows = np.asarray(road_rows)
    if road_rows.size == 0:
        raise ValueError("band_edges needs at least one road row")
    lo, hi = int(road_rows.min()), int(road_rows.max()) + 1
    if hi - lo < n_bands:
        raise ValueError(f"road row span {hi - lo} < n_bands {n_bands}")
    return np.linspace(lo, hi, n_bands + 1).astype(int)


def alloc_budget(weights: np.ndarray, total: int) -> np.ndarray:
    """Split `total` over bands ∝ weights, largest-remainder, never over capacity.

    Deterministic: floor-quotients first, the remainder goes to the bands with
    the largest leftover capacity in descending-weight order (stable ties).
    If total exceeds the summed capacity it is capped at that sum.
    """
    weights = np.asarray(weights, dtype=np.int64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("alloc_budget needs a non-empty 1-D weight vector")
    if (weights < 0).any() or total < 0:
        raise ValueError("total must be >= 0")
    alloc = np.zeros_like(weights)
    remaining = int(min(total, weights.sum()))
    for _ in range(weights.size + 1):
        free = weights - alloc
        eligible = free > 0
        if remaining <= 0 or not eligible.any():
            break
        ideal = weights * remaining / weights[eligible].sum()
        add = np.minimum(free, np.floor(ideal).astype(np.int64))
        if add.sum() == 0:  # allocate leftover units without subtracting twice
            left = remaining
            for b in np.argsort(-weights, kind="stable"):
                if left == 0:
                    break
                if free[b] > 0:
                    add[b] = 1
                    left -= 1
        alloc += add
        remaining -= int(add.sum())
    return alloc


def plan_centers(mask: np.ndarray, image_id: str,
                 cap: int = config.D_CAP_PER_IMAGE) -> list:
    """Up to `cap` patch centers (dicts band/row/col) for one image, seeded.

    Bands split the ROAD-mask row span ("Tiefenbänder über
    Road-Zeilen"); candidate centers come from allowed_region() (erosion
    reused from part_c) intersected with the PATCH//2-interior rectangle.
    That intersection is needed because cv2 erosion treats out-of-image
    pixels as neutral: near the right/bottom image border an eroded center
    can have < PATCH//2 columns of work image left, and its 128-px window
    would leave the frame. Deterministic per (mask, image_id): the RNG is
    seeded via patch_center_seed and each band's picks are re-sorted
    row-major, so neither image order nor dict ordering can leak in.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Expected a 2-D mask')
    if not mask.any() or min(mask.shape) < config.PATCH:
        return []
    road_rows = np.nonzero(mask.any(axis=1))[0]
    if np.ptp(road_rows) + 1 < config.N_BANDS:
        return []
    edges = band_edges(road_rows)
    rows, cols = np.nonzero(allowed_region(mask))
    r = config.PATCH // 2
    h, w = mask.shape[:2]
    interior = ((rows >= r) & (rows < h - r) & (cols >= r) & (cols < w - r))
    rows, cols = rows[interior], cols[interior]
    if rows.size == 0:
        return []
    bands = np.searchsorted(edges, rows, side="right") - 1
    weights = np.bincount(bands, minlength=config.N_BANDS)
    budget = alloc_budget(weights, cap)
    rng = np.random.default_rng(patch_center_seed(image_id))
    planned = []
    for band in range(config.N_BANDS):
        n_band = int(budget[band])
        if n_band == 0:
            continue
        in_band = np.nonzero(bands == band)[0]
        picks = rng.choice(in_band, size=n_band, replace=False)
        planned.extend({"band": int(band), "row": int(rows[i]), "col": int(cols[i])}
                       for i in sorted(picks))
    return planned


def shard_paths(stage: int) -> tuple:
    """(shard memmap path, index csv path) for a stage — both under outputs/patches/."""
    if stage not in config.D_EPOCHS:
        raise ValueError(f"unknown stage {stage!r}, expected one of {sorted(config.D_EPOCHS)}")
    return (config.PATCHES_DIR / f"patches_stage{stage}.npy",
            config.PATCHES_DIR / f"index_stage{stage}.csv")


def write_index(path, rows: list) -> None:
    """Write the patch index (one row per patch) as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_index(path) -> list:
    """Read the patch index back as typed dicts; FileNotFoundError hints at extract."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"missing patch index {path} — run: .venv/bin/python -m aibip.part_d extract")
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"patch_idx": int(row["patch_idx"]), "image_id": row["image_id"],
                         "band": int(row["band"]), "row": int(row["row"]),
                         "col": int(row["col"])})
    return rows


def reject_patch_label(label: str) -> bool:
    """True when a patch label should be excluded during Part-D extraction."""
    if not label:
        return False
    parts = set(label.split("+"))
    return bool(parts & set(config.D_EXCLUDE_PATCH_LABELS))


def split_ids(image_ids: list, val_frac: float = config.D_VAL_FRAC,
              seed: int = config.SEED) -> tuple:
    """Seeded train/val split BY IMAGE_ID (no patch-level leakage).

    Sorted unique ids, one seeded permutation, val = round(n * val_frac) (>= 1).
    Both returned lists are sorted, disjoint, and together cover every id.
    """
    ids = sorted(set(image_ids))
    if len(ids) < 2:
        raise ValueError(f"need >= 2 image ids for a split, got {len(ids)}")
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    n_val = min(len(ids)-1, max(1, round(len(ids) * val_frac)))
    perm = np.random.default_rng(seed).permutation(len(ids))
    val = [ids[i] for i in sorted(perm[:n_val])]
    train = [ids[i] for i in sorted(perm[n_val:])]
    return train, val


def select_rows(rows: list, image_ids: list) -> list:
    """Index rows whose image_id is in `image_ids` (order preserved = shard order)."""
    wanted = set(image_ids)
    selected = [row for row in rows if row["image_id"] in wanted]
    if not selected:
        raise ValueError("select_rows matched no index rows for the given ids")
    return selected


# --- extract (torch-free) ------------------------------------------------------

def _load_mask(path) -> np.ndarray:
    """Road-mask PNG as bool (grayscale; imgio.load_image would force 3 channels)."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cv2 could not read mask: {path}")
    return mask.astype(bool)


def _legacy_extract(stage: int = 1) -> None:
    """CLI body: sample patches over a stage's data dir -> uint8 memmap + index.

    skip-if-exists like part_c: only when BOTH shard and index exist is the
    pass skipped (delete to force). Patch centers are planned from the masks
    alone (pass 1), then the shard is filled with one preprocess() call per
    image (pass 2). (row, col) is the patch CENTER; PATCH is even, so the
    window is [row-64, row+64) x [col-64, col+64).
    """
    data_dir = config.D_DATA_DIR[stage]
    paths = imgio.list_images(data_dir)
    if not paths:
        raise ValueError(f"no images found in {data_dir}")
    shard_path, index_path = shard_paths(stage)
    if shard_path.exists() and index_path.exists():
        print(f"extract(D): {index_path.name} exists ({len(read_index(index_path))} patches) "
              f"— skip-if-exists, delete to force")
        return
    config.PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    planned, empty_ids = [], []
    rejected = {}
    r = config.PATCH // 2
    for path in paths:
        image_id = Path(path).stem
        mask_path = config.MASKS_DIR / f"{image_id}.png"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"missing road mask {mask_path} — run: "
                f".venv/bin/python -m aibip.part_c run {data_dir}")
        centers = plan_centers(_load_mask(mask_path), image_id)
        if centers and config.D_FILTER_PATCH_CONFOUNDERS:
            proc = preprocess(imgio.load_image(path))
            keep = []
            counts = {}
            for rec in centers:
                y, x = rec["row"], rec["col"]
                patch = proc[y - r:y + r, x - r:x + r]
                reason = patch_confounder_label(patch)
                if reject_patch_label(reason):
                    counts[reason] = counts.get(reason, 0) + 1
                else:
                    keep.append(rec)
            if counts:
                rejected[image_id] = counts
            centers = keep
        if not centers:
            empty_ids.append(image_id)
            continue
        planned.extend({"patch_idx": len(planned), "image_id": image_id, **c}
                       for c in centers)
    if not planned:
        raise ValueError("no valid patch centers in any mask — allowed regions all empty?")
    shard = np.lib.format.open_memmap(
        shard_path, mode="w+", dtype=np.uint8,
        shape=(len(planned), config.PATCH, config.PATCH, 3))
    by_image: dict = {}
    for rec in planned:
        by_image.setdefault(rec["image_id"], []).append(rec)
    for i, path in enumerate(paths, 1):
        recs = by_image.get(Path(path).stem)
        if not recs:
            continue
        proc = preprocess(imgio.load_image(path))
        h, w = proc.shape[:2]
        for rec in recs:
            y, x = rec["row"], rec["col"]
            if not (r <= y < h - r and r <= x < w - r):
                raise ValueError(f"center ({y},{x}) of {path} yields an out-of-bounds patch")
            shard[rec["patch_idx"]] = proc[y - r:y + r, x - r:x + r]
        if i % 25 == 0:
            print(f"extract(D): {i}/{len(paths)} images sampled")
    shard.flush()
    write_index(index_path, planned)
    per_band = np.bincount([rec["band"] for rec in planned], minlength=config.N_BANDS)
    print(f"extract(D): {len(planned)} patches from {len(paths) - len(empty_ids)} images "
          f"({len(empty_ids)} without valid centers) -> {shard_path.name}")
    print(f"  per band {per_band.tolist()} (0=far .. {config.N_BANDS - 1}=near)")
    if rejected:
        n_rejected = sum(sum(counts.values()) for counts in rejected.values())
        excluded = "/".join(sorted(config.D_EXCLUDE_PATCH_LABELS))
        print(f"  skipped {n_rejected} confounder patch candidates "
              f"from {len(rejected)} images ({excluded} filter)")


# --- torch section (lazy imports ONLY below this line) --------------------------

def _get_device():
    """MPS when available, CPU fallback (import torch HERE, not at module level)."""
    import torch
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model():
    """Denoising Conv-AE; returns the model (torch imported lazily).

    128x128x3 -> 4 stride-2 convs [32, 64, 128, 256] -> 8x8x256 bottleneck ->
    mirrored ConvTranspose decoder (kernel 4 to double spatial size cleanly),
    Sigmoid output for [0, 1] targets. ~1.08 M parameters — sized for the
    Stage-2 patch budget (~300k patches), small enough for minutes-scale
    dry runs (justification in the report).
    """
    import torch

    class ConvAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            channels = (32, 64, 128, 256)
            enc, cin = [], 3
            for cout in channels:
                enc += [torch.nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                        torch.nn.ReLU(inplace=True)]
                cin = cout
            dec = []
            for cout in reversed(channels[:-1]):
                dec += [torch.nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
                        torch.nn.ReLU(inplace=True)]
                cin = cout
            dec += [torch.nn.ConvTranspose2d(cin, 3, 4, stride=2, padding=1),
                    torch.nn.Sigmoid()]
            self.encoder = torch.nn.Sequential(*enc)
            self.decoder = torch.nn.Sequential(*dec)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return ConvAE()


def _make_dataset(shard_path, rows: list):
    """torch Dataset over the uint8 memmap shard, yielding CHW float32 in [0, 1].

    Reads patches lazily from the memmap per __getitem__ — no full float copy
    of the shard in RAM. `rows` (index subset) define membership, so the
    train/val split by image_id is enforced at the dataset level.
    """
    import torch
    if not rows:
        raise ValueError("_make_dataset needs at least one index row")

    class _ShardDataset(torch.utils.data.Dataset):
        def __init__(self, shard_path, rows):
            self.shard = np.load(shard_path, mmap_mode="r")
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            # np.array() copies: the memmap view is non-writable, torch would warn
            patch = np.array(self.shard[self.rows[i]["patch_idx"]])
            return torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0

    return _ShardDataset(shard_path, rows)


def _evaluate(model, loader, device) -> float:
    """Mean clean-reconstruction MSE over a loader (denoising eval: no noise)."""
    import torch
    loss_fn = torch.nn.MSELoss()
    total, seen = 0.0, 0
    model.eval()
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            total += float(loss_fn(model(x), x)) * x.size(0)
            seen += x.size(0)
    if seen == 0:
        raise ValueError("_evaluate got an empty loader")
    return total / seen


def _legacy_train(stage: int = 1) -> None:
    """CLI body: train the Conv-AE on a stage's shard + write all run artifacts.

    Determinism: torch/numpy seeded from SEED, DataLoader shuffle from a
    seeded generator, split via split_ids(seed). Artifacts under
    outputs/runs/stageN/: best.pt (lowest val loss incl. split metadata),
    history.csv, loss curve + recon grid (copies in docs/assets/ for the
    report). The recon grid ALWAYS includes the best- and worst-reconstructed
    patches of the whole stage set (submission list).
    """
    shard_path, index_path = shard_paths(stage)
    if not shard_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            f"missing shard/index in {config.PATCHES_DIR} — run: "
            f".venv/bin/python -m aibip.part_d extract --stage {stage}")
    import torch
    torch.manual_seed(config.SEED)
    rows = read_index(index_path)
    ids = sorted({row["image_id"] for row in rows})
    train_ids, val_ids = split_ids(ids)
    train_rows, val_rows = select_rows(rows, train_ids), select_rows(rows, val_ids)
    run_dir = config.RUNS_DIR / f"stage{stage}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = _get_device()
    model = _build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    generator = torch.Generator().manual_seed(config.SEED)
    train_loader = torch.utils.data.DataLoader(
        _make_dataset(shard_path, train_rows), batch_size=config.D_BATCH[stage],
        shuffle=True, generator=generator)
    val_loader = torch.utils.data.DataLoader(
        _make_dataset(shard_path, val_rows), batch_size=config.D_BATCH[stage],
        shuffle=False)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.D_LR)
    history, best_val = [], math.inf
    t0 = time.perf_counter()
    for epoch in range(1, config.D_EPOCHS[stage] + 1):
        model.train()
        running, seen = 0.0, 0
        for x in train_loader:
            x = x.to(device)
            noisy = (x + torch.randn_like(x) * config.D_NOISE_STD).clamp(0.0, 1.0)
            loss = loss_fn(model(noisy), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * x.size(0)
            seen += x.size(0)
        train_loss = running / seen
        val_loss = _evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            marker = " *"
            torch.save({"stage": stage, "epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "n_params": n_params,
                        "device": str(device), "seed": config.SEED,
                        "n_patches": len(rows), "n_train_patches": len(train_rows),
                        "n_val_patches": len(val_rows),
                        "train_ids": train_ids, "val_ids": val_ids,
                        "state_dict": model.state_dict()}, run_dir / "best.pt")
        print(f"epoch {epoch:3d}: train {train_loss:.5f}  val {val_loss:.5f}{marker}")
    wall = time.perf_counter() - t0
    _write_history(run_dir / "history.csv", history)
    _loss_curve(history, [run_dir / "loss_curve.png",
                          config.FIGURES_DIR / f"d_loss_curve_stage{stage}.png"])
    errors = _reconstruct_errors(model, shard_path, rows, device)
    order = np.argsort(errors)  # ascending: order[0] = best, order[-1] = worst
    rng = np.random.default_rng(config.SEED)
    middle = rng.choice(order[1:-1], size=min(4, max(order.size - 2, 0)), replace=False)
    picks = [int(order[-1])] + sorted(int(i) for i in middle) + [int(order[0])]
    shard = np.load(shard_path, mmap_mode="r")
    sheet = _build_sheet(shard, rows, picks, errors,
                         _reconstruct_patches(model, shard, picks, device))
    for out in (run_dir / "recon_grid.jpg",
                config.FIGURES_DIR / f"d_recon_grid_stage{stage}.png"):
        imgio.save_image(out, sheet)
    print(f"train(D): stage {stage} done in {wall:.1f} s on {device.type} — "
          f"{n_params} params, {len(train_ids)}/{len(val_ids)} train/val images "
          f"({len(train_rows)}/{len(val_rows)} patches), best val {best_val:.5f} "
          f"(worst patch err {float(errors[order[-1]]):.5f}, best {float(errors[order[0]]):.5f})")


def _write_history(path, history: list) -> None:
    """Loss history (epoch/train/val) as CSV next to the checkpoint."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def _loss_curve(history: list, out_paths: list) -> None:
    """Train+val MSE per epoch, best epoch marked; written to every out_path."""
    import matplotlib.pyplot as plt
    if not history:
        raise ValueError("_loss_curve needs at least one epoch")
    epochs = [row["epoch"] for row in history]
    best = int(np.argmin([row["val_loss"] for row in history]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [row["train_loss"] for row in history], label="train (denoised input)")
    ax.plot(epochs, [row["val_loss"] for row in history], label="val (clean input)")
    ax.axvline(epochs[best], color="grey", ls=":", lw=1,
               label=f"best val @ epoch {epochs[best]}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Part D: Conv-AE reconstruction loss (Stage 1 mini dry run)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for out in out_paths:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
    plt.close(fig)


def _reconstruct_errors(model, shard_path, rows: list, device,
                        batch_size: int = 256) -> np.ndarray:
    """Per-patch clean MSE (model vs. input) over the given index rows."""
    import torch
    loader = torch.utils.data.DataLoader(
        _make_dataset(shard_path, rows), batch_size=batch_size, shuffle=False)
    errors = np.empty(len(rows), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        pos = 0
        for x in loader:
            x = x.to(device)
            per_patch = ((model(x) - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            errors[pos:pos + per_patch.size] = per_patch
            pos += per_patch.size
    if pos != len(rows):
        raise ValueError(f"_reconstruct_errors covered {pos} of {len(rows)} rows")
    return errors


def _reconstruct_patches(model, shard, indices: list, device) -> np.ndarray:
    """Reconstructed patches (float [0,1], HWC) for the given shard indices."""
    import torch
    batch = np.stack([np.ascontiguousarray(shard[i]) for i in indices])
    x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x).clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()
    return out


def _build_sheet(shard, rows: list, picks: list, errors: np.ndarray,
                 recon01: np.ndarray) -> np.ndarray:
    """Contact sheet: per pick one row (original | reconstruction), labeled.

    Labels carry image_id, depth band and per-patch MSE (compact on purpose:
    the tile is only 256 px wide). The picks order is the caller's — train
    uses worst -> ... -> best so the two extremes are the outer rows
    ("best- and worst-reconstructed patches").
    """
    tiles = []
    for k, j in enumerate(picks):
        rec = rows[j]
        orig = np.ascontiguousarray(shard[rec["patch_idx"]])
        tile = np.hstack([orig, (recon01[k] * 255.0).round().astype(np.uint8)])
        _label(tile, (4, 18), f"{rec['image_id']} b{rec['band']} {errors[j]:.4f}")
        tiles.append(tile)
    return imgio.montage(tiles, cols=1, pad=4)


def _legacy_preview(stage: int = 1, n: int = 8, seed: int = config.SEED) -> None:
    """CLI body: QC the trained checkpoint WITHOUT retraining.

    Loads outputs/runs/stageN/best.pt, computes the clean per-patch MSE over
    ALL of the stage's patches, prints per-band means (the perspective-QC
    signal: far bands should NOT dominate), and writes a seeded
    original-vs-reconstruction sheet to outputs/qc/.
    """
    shard_path, index_path = shard_paths(stage)
    ckpt_path = config.RUNS_DIR / f"stage{stage}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"missing checkpoint {ckpt_path} — run: "
            f".venv/bin/python -m aibip.part_d train --stage {stage}")
    rows = read_index(index_path)
    import torch
    device = _get_device()
    model = _build_model().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    errors = _reconstruct_errors(model, shard_path, rows, device)
    bands = np.asarray([row["band"] for row in rows])
    for band in sorted(set(bands.tolist())):
        print(f"  band {band}: mean recon error {float(errors[bands == band].mean()):.5f}")
    picks = sorted(int(i) for i in np.random.default_rng(seed)
                   .choice(len(rows), size=min(n, len(rows)), replace=False))
    shard = np.load(shard_path, mmap_mode="r")
    qc_dir = config.OUTPUTS_DIR / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    out_path = qc_dir / f"d_preview_stage{stage}.jpg"
    imgio.save_image(out_path, _build_sheet(
        shard, rows, picks, errors, _reconstruct_patches(model, shard, picks, device)))
    print(f"preview(D): wrote {out_path} ({len(picks)} seeded patches, "
          f"checkpoint from epoch {checkpoint['epoch']})")


# The default D commands now use the auditable Mini-only implementation.
# Existing low-level model helpers remain compatible with the historical E code.
from aibip.d_data import extract
from aibip.d_training import train, preview, main

if __name__ == '__main__':
    main()