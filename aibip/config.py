"""Constants-only configuration. No classes, no YAML, no env parsing."""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
DATA_ROOT = Path("data")
MINI_SOURCE_DIR = DATA_ROOT / "26-07-13-Cyclomedia-Test-Hammersbach"  # 1765 JPGs, source for the mini set
MINI_DIR = DATA_ROOT / "mini"  # official 100-image mini set, generated deterministically in a later task
POOL_DIR = DATA_ROOT / "26-07-14 Cyclomedia-Test-Hammersbach" / "panoramas"  # 12432 JPGs, Stage-2 training pool
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
MASKS_DIR = OUTPUTS_DIR / "masks"
PATCHES_DIR = OUTPUTS_DIR / "patches"
RUNS_DIR = OUTPUTS_DIR / "runs"
FIGURES_DIR = Path("docs") / "assets"
FLAGS_CSV = OUTPUTS_DIR / "flags.csv"


# --- Image / processing ------------------------------------------------------
# (height, width) full-frame reference resolution for previews/figures only
# (SAM resizes to 1024^2 internally, so full-frame res is a pure display choice).
WORK_RES = (1152, 1536)
RAW_RES = (6144, 8192)  # (height, width)

# Fixed ROI crop, measured on grid/strip montages over 100 mini images
# (not guessed), then deliberately relaxed by review:
ROI_TOP_FRAC = 0.40
# Sky cut: horizon median at 0.418 H, lowest observed ground line ~0.40 H.
# 0.40 sits exactly ON that line: maximal sky removal, zero margin — the
# farthest road in the most open views touches the cut (verified on montages).
ROI_BOTTOM_FRAC = 0.8
# Hood cut, deliberately set ABOVE the measured clean line (worst hood crest
# is ~0.73 H in the WE7EJ framing regime; the fully clean cut would be
# 0.71-0.73). Trade-off accepted by review: keeps more near-field road at the
# cost of hood/board slivers in most WE7EJ-style images. flags.csv can NOT
# catch these (majority effect vs median/MAD — see report); Part C's PATCH/2
# mask erosion (~64 px at the bottom border) trims the slivers out of Part D's
# allowed region.
WORK_WIDTH = 2048
# preprocess() output width. The [0.40, 0.80] band holds ~2458 of 6144
# rows -> 614 rows at width 2048, enough for N_BANDS depth bands of one
# PATCH-height each.
# WORK_WIDTH = 3072  # documented fallback if Stage 1 shows cracks are invisible (~830 rows, ~2x slower)

PATCH = 128  # patch size in px fed to the Conv-AE
N_BANDS = 4  # depth bands over road rows for perspective-aware sampling/scoring
USE_CLAHE = True  # CLAHE on L channel in preprocessing
N_MINI = 100  # official mini-set size
# Single global seed; mini-set sampling and patch RNG derive from it,
# per-image derivations use stable hashes.
SEED = 42

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}
