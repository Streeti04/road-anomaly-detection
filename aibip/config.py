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
USE_CLAHE = True  # CLAHE on L channel in preprocessing; A/B figure documents the rationale
N_MINI = 100  # official mini-set size
# Single global seed; mini-set sampling and patch RNG derive from it,
# per-image derivations use stable hashes.
SEED = 42

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# --- Part C (Road segmentation) ---------------------------------------------
C_BACKEND = "mobile_sam"  # "mobile_sam" default; "sam" optional full-SAM test path
C_PROMPT_PROFILE = "basic"  # simple point prompts first; no side/box over-steering
C_ROAD_CORE_HALF_TOP_FRAC = 0.10  # keep a center corridor after SAM; top rows are narrow
C_ROAD_CORE_HALF_BOTTOM_FRAC = 0.22  # bottom rows may contain wider near-field road
C_EXPAND_HALF_TOP_FRAC = 0.06
C_EXPAND_HALF_BOTTOM_FRAC = 0.16
C_ASPHALT_SAT_MIN = 20
C_ASPHALT_LOW_SAT_VAL_MAX = 130  # low-saturation pixels are accepted only when not bright sidewalk
C_ASPHALT_VAL_MAX = 170
C_ROAD_MAX_HEIGHT_FRAC = 0.22

# --- Part D (Conv-AE on road patches) -----------------------------------------
# Stage 1 = dry run on data/mini, Stage 2 = pool run; same code paths,
# stage-dependent values are looked up, not branched.
D_DATA_DIR = {1: MINI_DIR, 2: POOL_DIR}  # raw images per stage
D_CAP_PER_IMAGE = 24  # max patches per image, budget split ∝ allowed area per band
D_VAL_FRAC = 0.10  # validation split BY IMAGE_ID, no patch-level leakage
D_EPOCHS = {1: 30, 2: 20}  # dry run: minutes-scale proof of concept; pool: <= 2 h budget
D_BATCH = {1: 64, 2: 256}  # small stage-1 batches (MPS risk); 256 for stage 2
D_LR = 1e-3  # Adam learning rate
D_NOISE_STD = 0.02  # denoising AE: input + N(0, 0.02) on [0,1] floats
D_FILTER_PATCH_CONFOUNDERS = True  # skip only configured non-road labels during shard extraction
D_EXCLUDE_PATCH_LABELS = {"vegetation"}  # keep markings/shadows for training variety; E handles ranking
