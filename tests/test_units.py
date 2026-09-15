"""Pure-function unit tests. No models, no data directory, no GPU."""

import csv
from pathlib import Path

import numpy as np
import pytest

from aibip import config, imgio, mini, part_b
from aibip.patch_filter import patch_confounder_label
from aibip.part_b import preprocess
from aibip.part_c import segment_road


def test_resize_shapes():
    rng = np.random.default_rng(config.SEED)
    for res in [(64, 96), (200, 150)]:
        image = rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8)
        out = imgio.resize(image, res)
        assert out.shape == (res[0], res[1], 3)


def test_to_work_resolution():
    rng = np.random.default_rng(config.SEED)
    image = rng.integers(0, 255, size=(6144, 8192, 3), dtype=np.uint8)
    out = imgio.to_work_resolution(image)
    assert out.shape == (1152, 1536, 3)
    assert out.dtype == np.uint8


def test_resize_rejects_empty():
    with pytest.raises(ValueError):
        imgio.resize(np.empty((0, 0, 3), dtype=np.uint8))


def test_montage_grid_shape():
    rng = np.random.default_rng(config.SEED)
    images = [rng.integers(0, 255, size=(32, 48, 3), dtype=np.uint8) for _ in range(6)]
    cols, pad = 4, 4
    out = imgio.montage(images, cols=cols, pad=pad)
    rows = -(-len(images) // cols)  # ceil
    assert out.shape == (rows * (32 + 2 * pad), cols * (48 + 2 * pad), 3)
    assert out.dtype == np.uint8


def test_montage_empty_raises():
    with pytest.raises(ValueError):
        imgio.montage([])


def test_montage_resizes_mixed_sizes():
    rng = np.random.default_rng(config.SEED)
    small = rng.integers(0, 255, size=(32, 48, 3), dtype=np.uint8)
    large = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
    out = imgio.montage([small, large], cols=2, pad=2)
    # 1 row x 2 cols; both tiles resized to the first image's size
    assert out.shape == (32 + 2 * 2, 2 * (48 + 2 * 2), 3)


def test_load_save_roundtrip(tmp_path):
    rng = np.random.default_rng(config.SEED)
    image = rng.integers(0, 255, size=(40, 60, 3), dtype=np.uint8)
    path = tmp_path / "x.png"
    imgio.save_image(path, image)
    loaded = imgio.load_image(path)
    assert loaded.shape == image.shape
    assert loaded.dtype == image.dtype
    assert np.array_equal(loaded, image)


def test_load_image_missing_raises():
    with pytest.raises(FileNotFoundError):
        imgio.load_image(Path("does/not/exist.png"))


def test_list_images_sorted_and_filtered(tmp_path):
    for name in ["b.jpg", "a.JPG", "c.txt", "d.jpeg"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "e.png").write_bytes(b"")
    # plain sorted() on Path objects: case-sensitive string sort
    assert imgio.list_images(tmp_path) == [
        tmp_path / "a.JPG", tmp_path / "b.jpg", tmp_path / "d.jpeg"
    ]


def test_config_sanity():
    for res in (config.WORK_RES, config.RAW_RES):
        assert res[1] / res[0] == pytest.approx(4 / 3, abs=1e-3)
    assert config.PATCH >= 64 and (config.PATCH & (config.PATCH - 1)) == 0
    assert config.N_BANDS == 4
    assert config.N_MINI == 100
    assert config.USE_CLAHE is True
    assert isinstance(config.SEED, int)


def test_stubs_importable_and_raise():
    # Import gate since T2: both pipeline entry points are implemented
    # (the old stub NotImplementedError check was retired with T2).
    assert callable(preprocess)
    assert callable(segment_road)


def _make_fake_source(tmp_path: Path, n: int = 10) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for i in range(n):
        (source / f"img_{i:02d}.jpg").write_bytes(b"\xff\xd8fake")
    return source


def test_make_mini_selects_and_links(tmp_path):
    source = _make_fake_source(tmp_path)
    dest = tmp_path / "mini"
    links = mini.make_mini(n=4, seed=0, source=source, dest=dest)
    assert len(links) == 4
    assert all(link.exists() for link in links)
    assert sorted(dest.iterdir()) == sorted(links)


def test_make_mini_deterministic(tmp_path):
    source = _make_fake_source(tmp_path)
    first = mini.make_mini(n=4, seed=0, source=source, dest=tmp_path / "mini_a")
    second = mini.make_mini(n=4, seed=0, source=source, dest=tmp_path / "mini_b")
    assert [p.name for p in first] == [p.name for p in second]


def test_make_mini_idempotent(tmp_path):
    source = _make_fake_source(tmp_path)
    dest = tmp_path / "mini"
    first = mini.make_mini(n=4, seed=0, source=source, dest=dest)
    second = mini.make_mini(n=4, seed=0, source=source, dest=dest)
    assert len(list(dest.iterdir())) == 4
    assert len(second) == 4
    assert first == second


# --- Part B (T1) --------------------------------------------------------------

_BIG = np.random.default_rng(0).integers(0, 255, size=(6144, 8192, 3), dtype=np.uint8)


def test_crop_roi_frac_bounds():
    image = np.zeros((1000, 10, 3), dtype=np.uint8)
    band = part_b.crop_roi(image)
    assert band.shape == (800 - 400, 10, 3)
    with pytest.raises(ValueError):
        part_b.crop_roi(np.zeros((1, 10, 3), dtype=np.uint8))


def test_resize_band_aspect():
    band = part_b.crop_roi(_BIG)
    out = part_b.resize_band(band)
    # fixed expectation from the raw-resolution band (4915-2457 = 2458 x 8192) at WORK_WIDTH
    assert out.shape == (614, config.WORK_WIDTH, 3)
    # and the same value via the independent formula the function implements
    expected_h = max(1, round(band.shape[0] * config.WORK_WIDTH / band.shape[1]))
    assert out.shape == (expected_h, config.WORK_WIDTH, 3)


def test_preprocess_shape_and_dtype():
    out = preprocess(_BIG)
    assert out.shape == (614, config.WORK_WIDTH, 3)
    assert out.dtype == np.uint8


def test_preprocess_clahe_switch(monkeypatch):
    monkeypatch.setattr(config, "USE_CLAHE", False)
    without = preprocess(_BIG)
    assert np.array_equal(without, part_b.resize_band(part_b.crop_roi(_BIG)))
    monkeypatch.setattr(config, "USE_CLAHE", True)
    with_clahe = preprocess(_BIG)
    assert not np.array_equal(with_clahe, without)


def test_flag_outliers_plants_outlier():
    rng = np.random.default_rng(config.SEED)
    records = [{"image_id": f"norm_{i:02d}", "v": float(v), "s": 10.0}
               for i, v in enumerate(rng.integers(126, 131, size=20))]
    records.append({"image_id": "dark_hood", "v": 40.0, "s": 10.0})
    rows = part_b.flag_outliers(records)
    assert len(rows) == 1
    assert rows[0]["image_id"] == "dark_hood"
    assert rows[0]["metric"] == "v"
    assert abs(rows[0]["z"]) > 3
    # identical constant records -> MAD == 0 -> no flags at all
    constant = [{"image_id": f"c{i}", "v": 100.0, "s": 50.0} for i in range(10)]
    assert part_b.flag_outliers(constant) == []


def test_hood_strip_stats_synthetic():
    image = np.full((6144, 16, 3), 200, dtype=np.uint8)
    stats = part_b.hood_strip_stats(image)
    assert stats["v"] == pytest.approx(200.0, abs=1.0)
    assert stats["s"] == pytest.approx(0.0, abs=1.0)
    with pytest.raises(ValueError):
        part_b.hood_strip_stats(np.zeros((1, 16, 3), dtype=np.uint8))


def test_write_flags_preserves_other_stages(tmp_path, monkeypatch):
    csv_path = tmp_path / "flags.csv"
    monkeypatch.setattr(config, "FLAGS_CSV", csv_path)
    csv_path.write_text(
        "image_id,stage,reason,metric,value,z\nimg_x,C,area_outlier,area,0.9,3.5\n")
    b_row = {"image_id": "img_b", "stage": "B", "reason": "hood_strip_outlier",
             "metric": "v", "value": 40.0, "z": -5.2}
    part_b.write_flags([b_row])
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert any(r["stage"] == "C" and r["image_id"] == "img_x" for r in rows)
    assert any(r["stage"] == "B" and r["image_id"] == "img_b" for r in rows)
    # rewrite semantics: a second write must not duplicate the B row
    part_b.write_flags([b_row])
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert sum(r["stage"] == "B" for r in rows) == 1
    assert sum(r["stage"] == "C" for r in rows) == 1


# --- Part C (T2) ----------------------------------------------------------------

import importlib  # noqa: E402  (append-only section)
import sys  # noqa: E402

part_c = importlib.import_module("aibip.part_c")


def test_select_candidate_prefers_center_over_side_spill():
    h, w = 100, 100
    side_spill = np.zeros((h, w), dtype=bool)
    side_spill[45:90, 35:100] = True    # center hit, but large right-side grass merge
    road = np.zeros((h, w), dtype=bool)
    road[45:90, 42:58] = True           # clean center-road candidate
    tiny = np.zeros((h, w), dtype=bool)
    tiny[78:82, 48:52] = True           # anchor hit, too little center coverage
    masks = np.stack([side_spill, road, tiny])
    scores = np.array([0.9, 0.4, 0.99]) # raw SAM score must not decide this
    mask, index = part_c.select_candidate(masks, scores, h, w)
    assert index == 1
    assert np.array_equal(mask, road)


def test_select_candidate_prefers_more_center_hits():
    h, w = 100, 100
    masks = np.zeros((3, h, w), dtype=bool)
    masks[0, 78:82, 48:52] = True       # only near anchor
    masks[1, 60:90, 42:58] = True       # mid + near anchors
    masks[2, 45:90, 44:56] = True       # far + mid + near anchors
    scores = np.array([0.99, 0.5, 0.1])
    mask, index = part_c.select_candidate(masks, scores, h, w)
    assert index == 2
    # deterministic: a second identical call must return the same candidate
    assert np.array_equal(part_c.select_candidate(masks, scores, h, w)[0], mask)


def test_road_corridor_removes_side_grass_merge():
    mask = np.zeros((120, 200), dtype=bool)
    mask[20:110, 20:190] = True         # road merged with right grass shoulder
    corridor = part_c.road_corridor(mask)
    assert corridor[80, 100]            # center road remains
    assert not corridor[80, 185]        # far side spill removed
    assert corridor.sum() < mask.sum()


def test_select_candidate_with_image_rejects_thin_line():
    h, w = 120, 200
    proc = np.full((h, w, 3), 95, dtype=np.uint8)
    thin = np.zeros((h, w), dtype=bool)
    for y in range(35, 110):
        x = 45 + y // 3
        thin[y, x:x + 5] = True
    road = np.zeros((h, w), dtype=bool)
    road[35:110, 65:135] = True
    masks = np.stack([thin, road])
    scores = np.array([0.99, 0.1])
    assert part_c.select_candidate_with_image(masks, scores, proc) == 1


def test_expand_road_surface_grows_asphalt_not_grass():
    h, w = 120, 200
    proc = np.zeros((h, w, 3), dtype=np.uint8)
    proc[:] = (40, 120, 40)              # vegetation background
    proc[30:110, 60:140] = (95, 95, 95)  # asphalt-like road surface
    seed = np.zeros((h, w), dtype=bool)
    seed[40:105, 92:108] = True          # cautious MobileSAM center strip

    expanded = part_c.expand_road_surface(proc, seed)

    assert expanded[80, 70]
    assert expanded[80, 130]
    assert not expanded[80, 30]
    assert not expanded[80, 170]
    assert expanded.sum() > seed.sum()


def test_allowed_region_border_clearance():
    r = config.PATCH // 2    # 64
    mask = np.zeros((614, 2048), dtype=bool)
    mask[100:500, 200:1800] = True
    region = part_c.allowed_region(mask)
    ys, xs = np.nonzero(region)
    # every True pixel sits >= PATCH//2 inside the rectangle border
    assert ys.min() >= 100 + r and ys.max() <= 499 - r
    assert xs.min() >= 200 + r and xs.max() <= 1799 - r
    # probes: clearance of exactly r survives, r-1 and deep-edge do not
    assert region[100 + r, 1024]
    assert not region[100 + r - 1, 1024]
    assert not region[100 + 10, 1024]
    # strict subset: the rectangle corners are eroded away
    assert not region[100, 200] and not region[499, 1799]
    assert region.sum() < mask.sum()


def test_clean_mask_picks_largest_component():
    mask = np.zeros((614, 2048), dtype=bool)
    mask[100:300, 200:400] = True     # big 200x200 blob
    mask[400:420, 1000:1020] = True   # small 20x20 blob far away (survives open)
    mask[50:53, 1500:1503] = True     # speckle
    mask[500, 60] = True              # single-pixel speckle
    cleaned = part_c.clean_mask(mask)
    assert cleaned[200, 300]          # big blob survives
    assert not cleaned[410, 1010]     # small blob removed: not the largest
    assert not cleaned[51, 1501] and not cleaned[500, 60]
    assert cleaned.sum() < 200 * 200  # open shaves the border, no satellites


def test_mask_stats_values():
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:50] = True                # 20% area, centroid row 39.5
    stats = part_c.mask_stats(mask, 0.75)
    assert stats["area_frac"] == pytest.approx(0.20)
    assert stats["centroid_row_frac"] == pytest.approx(0.395)
    assert stats["strip_cov"] == pytest.approx(0.0)  # rows [70, 85) untouched
    assert stats["sam_score"] == pytest.approx(0.75)
    assert stats["n_components"] == 1
    empty = part_c.mask_stats(np.zeros((100, 100), dtype=bool), 0.0)
    assert empty["area_frac"] == 0.0
    assert empty["centroid_row_frac"] == 0.0
    assert empty["n_components"] == 0


def test_part_c_still_lazy():
    part_c_mod = importlib.import_module("aibip.part_c")
    assert callable(part_c_mod.segment_road)
    # lazy singleton: importing the module must NOT have built the predictor
    assert part_c_mod.__dict__.get("_PREDICTORS") == {}
    # and no model backend may have been pulled in by the import chain
    assert "torch" not in sys.modules
    assert "mobile_sam" not in sys.modules


# --------------------------------------------------------------- fetch_models

def test_fetch_sha256_of(tmp_path):
    import hashlib
    from aibip import fetch_models

    f = tmp_path / "blob.bin"
    f.write_bytes(b"aibip" * 1000)
    assert fetch_models.sha256_of(f) == hashlib.sha256(b"aibip" * 1000).hexdigest()


def test_fetch_models_download_then_idempotent(tmp_path, monkeypatch, capsys):
    from aibip import fetch_models

    src = tmp_path / "src.bin"
    src.write_bytes(b"weights" * 512)
    digest = fetch_models.sha256_of(src)

    models_dir = tmp_path / "models"
    # fetch_models binds MODELS_DIR at import time, so patch there, not in config
    monkeypatch.setattr(fetch_models, "MODELS_DIR", models_dir)
    monkeypatch.setattr(
        fetch_models, "SOURCES", {"fake.bin": (src.as_uri(), digest, src.stat().st_size)}
    )

    assert fetch_models.main() == 0            # first run: downloads
    assert (models_dir / "fake.bin").read_bytes() == src.read_bytes()
    assert not (models_dir / "fake.bin.part").exists()
    assert fetch_models.main() == 0            # second run: skips
    assert "already present" in capsys.readouterr().out


def test_fetch_models_rejects_bad_checksum(tmp_path, monkeypatch):
    from aibip import fetch_models

    src = tmp_path / "src.bin"
    src.write_bytes(b"corrupt")
    models_dir = tmp_path / "models"
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(
        fetch_models, "SOURCES", {"bad.bin": (src.as_uri(), "0" * 64, 7)}
    )

    with pytest.raises(SystemExit, match="checksum mismatch"):
        fetch_models.main()
    assert not (models_dir / "bad.bin").exists()      # nothing installed
    assert not (models_dir / "bad.bin.part").exists() # temp cleaned up


# --- Part D, Stage 1 ---------------------------------------------------------
# torch-free pure functions only: model + training loop are verified by running
# the actual documented dry run, never under pytest.

part_d = importlib.import_module("aibip.part_d")
part_e = importlib.import_module("aibip.part_e")


def test_part_d_patch_center_seed_stable():
    seed = part_d.patch_center_seed("WE1MY5C5")
    assert seed == part_d.patch_center_seed("WE1MY5C5")   # stable across calls
    assert seed != part_d.patch_center_seed("WE7EJMKD")   # differs per image
    assert 0 <= seed < 2 ** 32                            # crc32 range
    with pytest.raises(ValueError):
        part_d.patch_center_seed("")


def test_part_d_band_edges_split_span():
    edges = part_d.band_edges(np.array([100, 104, 500]))
    assert edges.tolist() == [100, 200, 300, 400, 501]    # last edge = max + 1
    band = np.searchsorted(edges, 500, side="right") - 1
    assert band == config.N_BANDS - 1                     # deepest road row in band 3
    with pytest.raises(ValueError):
        part_d.band_edges(np.array([], dtype=int))
    with pytest.raises(ValueError):
        part_d.band_edges(np.array([10, 11, 12]))         # span 2 < n_bands


def test_part_d_alloc_budget_proportional_and_capped():
    weights = np.array([600, 300, 90, 10])
    alloc = part_d.alloc_budget(weights, 24)
    assert alloc.sum() == 24
    assert alloc[0] >= alloc[1] >= alloc[2] >= alloc[3]   # monotone with weight
    assert np.all(part_d.alloc_budget(weights, 24) == alloc)  # deterministic
    assert np.all(part_d.alloc_budget(weights, 10_000) == weights)  # capped at capacity
    zero_band = part_d.alloc_budget(np.array([3, 1000, 0, 900]), 24)
    assert zero_band[2] == 0 and zero_band.sum() == 24    # empty band stays empty
    big = part_d.alloc_budget(np.array([10, 1000, 1000]), 10_000)
    assert np.all(big <= np.array([10, 1000, 1000])) and big.sum() == 2010
    with pytest.raises(ValueError):
        part_d.alloc_budget(np.array([], dtype=int), 5)
    with pytest.raises(ValueError):
        part_d.alloc_budget(weights, -1)


def test_part_d_plan_centers_inside_allowed_region():
    mask = np.zeros((614, 2048), dtype=bool)
    mask[100:500, 200:1800] = True
    centers = part_d.plan_centers(mask, "TESTID")
    assert 0 < len(centers) <= config.D_CAP_PER_IMAGE
    assert sorted({c["band"] for c in centers}) == [0, 1, 2, 3]  # all bands used
    region = part_c.allowed_region(mask)
    r = config.PATCH // 2
    for c in centers:
        assert 0 <= c["band"] < config.N_BANDS
        assert region[c["row"], c["col"]]                 # center in allowed region ...
        assert 100 + r <= c["row"] < 500 - r              # ... so the patch is fully road
        assert 200 + r <= c["col"] < 1800 - r
    assert centers == part_d.plan_centers(mask, "TESTID")  # deterministic per image_id
    assert centers != part_d.plan_centers(mask, "OTHER")    # and seeded per image_id
    # a road region touching the right image border must still yield in-frame
    # patches (erosion alone would allow centers with < PATCH//2 columns left)
    border_mask = np.zeros((614, 2048), dtype=bool)
    border_mask[100:500, 1500:2048] = True
    for c in part_d.plan_centers(border_mask, "BORDER"):
        assert config.PATCH // 2 <= c["col"] < 2048 - config.PATCH // 2


def test_part_d_split_by_image_id():
    ids = [f"WE{i:04d}" for i in range(100)]
    train, val = part_d.split_ids(ids)
    assert len(val) == 10 and len(train) == 90
    assert set(train) | set(val) == set(ids)
    assert not set(train) & set(val)                      # no id on both sides
    assert train == sorted(train) and val == sorted(val)
    assert (train, val) == part_d.split_ids(ids)          # deterministic under SEED
    with pytest.raises(ValueError):
        part_d.split_ids(["only_one"])
    with pytest.raises(ValueError):
        part_d.split_ids(ids, val_frac=1.0)


def test_part_d_select_rows_partition_no_leakage():
    rows = [{"patch_idx": i, "image_id": f"img{i % 4}", "band": 0, "row": i, "col": i}
            for i in range(8)]
    train_rows = part_d.select_rows(rows, ["img0", "img1"])
    val_rows = part_d.select_rows(rows, ["img2", "img3"])
    assert [r["patch_idx"] for r in train_rows] == [0, 1, 4, 5]
    assert [r["patch_idx"] for r in val_rows] == [2, 3, 6, 7]
    assert not {r["image_id"] for r in train_rows} & {r["image_id"] for r in val_rows}
    with pytest.raises(ValueError):
        part_d.select_rows(rows, ["missing_id"])


def test_part_d_index_roundtrip(tmp_path):
    rows = [{"patch_idx": 0, "image_id": "A", "band": 0, "row": 100, "col": 200},
            {"patch_idx": 1, "image_id": "B", "band": 3, "row": 400, "col": 1800}]
    path = tmp_path / "index.csv"
    part_d.write_index(path, rows)
    assert part_d.read_index(path) == rows                # typed roundtrip
    with pytest.raises(FileNotFoundError):
        part_d.read_index(tmp_path / "missing.csv")


def test_part_d_rejects_only_configured_patch_labels():
    assert part_d.reject_patch_label("") is False
    assert part_d.reject_patch_label("marking") is False
    assert part_d.reject_patch_label("shadow") is False
    assert part_d.reject_patch_label("vegetation") is True
    assert part_d.reject_patch_label("vegetation+marking") is True


def test_part_d_shard_memmap_roundtrip(tmp_path):
    # mirrors the extract() usage of np.lib.format.open_memmap (cache)
    shard_path = tmp_path / "shard.npy"
    data = np.random.default_rng(config.SEED).integers(
        0, 255, size=(4, 16, 16, 3), dtype=np.uint8)
    shard = np.lib.format.open_memmap(
        shard_path, mode="w+", dtype=np.uint8, shape=data.shape)
    shard[:] = data
    shard.flush()
    del shard
    loaded = np.load(shard_path, mmap_mode="r")
    assert loaded.dtype == np.uint8 and loaded.shape == data.shape
    assert np.array_equal(np.asarray(loaded[:]), data)


def test_part_d_still_lazy():
    part_d_mod = importlib.import_module("aibip.part_d")
    assert callable(part_d_mod.extract)
    # importing part_d (via part_b/part_c helpers) must not pull in torch
    assert "torch" not in sys.modules


# --- Part E (T5) --------------------------------------------------------------

def test_part_e_robust_band_stats_and_z_scores():
    errors = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    bands = np.array([0, 0, 0, 1, 1, 1])
    stats = part_e.robust_band_stats(errors, bands, n_bands=3)
    assert sorted(stats) == [0, 1]
    assert stats[0]["median"] == pytest.approx(2.0)
    assert stats[1]["median"] == pytest.approx(11.0)
    z = part_e.band_z_scores(np.array([2.0, 11.0]), np.array([0, 1]), stats)
    assert np.allclose(z, [0.0, 0.0])
    high = part_e.band_z_scores(np.array([3.4826]), np.array([0]), stats)
    assert high[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        part_e.robust_band_stats(np.array([1.0]), np.array([0, 1]))


def test_part_e_robust_band_stats_skips_tiny_bands():
    errors = np.array([1.0, 2.0, 10.0, 11.0, 12.0])
    bands = np.array([0, 0, 1, 1, 1])

    stats = part_e.robust_band_stats(errors, bands, n_bands=2, min_count=3)
    z = part_e.band_z_scores(np.array([2.0]), np.array([0]), stats)

    assert sorted(stats) == [1]
    assert abs(float(z[0])) < 2.0


def test_part_e_robust_band_stats_applies_scale_floor():
    stats = part_e.robust_band_stats(
        np.array([1.0, 1.0, 1.0]), np.array([0, 0, 0]),
        min_scale=0.25)

    assert stats[0]["scale"] == pytest.approx(0.25)


def test_part_e_top_fraction_mean():
    values = np.array([1.0, 2.0, 3.0, 100.0])
    assert part_e.top_fraction_mean(values, frac=0.25) == pytest.approx(100.0)
    assert part_e.top_fraction_mean(values, frac=0.50) == pytest.approx(51.5)
    with pytest.raises(ValueError):
        part_e.top_fraction_mean(np.array([]))
    with pytest.raises(ValueError):
        part_e.top_fraction_mean(values, frac=0.0)


def test_part_e_aggregate_image_scores_ranks_by_p90_z():
    rows = [
        {"patch_idx": 0, "image_id": "A", "band": 0, "row": 10, "col": 20},
        {"patch_idx": 1, "image_id": "A", "band": 0, "row": 11, "col": 21},
        {"patch_idx": 2, "image_id": "B", "band": 1, "row": 12, "col": 22},
        {"patch_idx": 3, "image_id": "B", "band": 1, "row": 13, "col": 23},
        {"patch_idx": 4, "image_id": "A", "band": 0, "row": 14, "col": 24},
        {"patch_idx": 5, "image_id": "A", "band": 0, "row": 15, "col": 25},
        {"patch_idx": 6, "image_id": "B", "band": 1, "row": 16, "col": 26},
        {"patch_idx": 7, "image_id": "B", "band": 1, "row": 17, "col": 27},
    ]
    errors = np.array([0.1, 0.2, 0.5, 0.6, 0.15, 0.25, 0.55, 0.65])
    z_scores = np.array([0.0, 0.5, 1.0, 3.0, 0.2, 0.3, 2.0, 2.0])
    records = part_e.aggregate_image_scores(
        rows, errors, z_scores, model_stage=1, analysis_stage=1, baseline_stage=1)
    assert [r["image_id"] for r in records] == ["A", "B"]
    assert [r["rank"] for r in records] == [1, 2]
    assert records[1]["score_status"] == "ok"
    assert records[1]["worst_patch_idx"] == 3
    assert records[1]["condition_score"] == pytest.approx(3.0)
    assert records[1]["score_mean_z"] == pytest.approx(2.0)


def test_part_e_aggregate_can_exclude_confounder_patches():
    rows = [
        {"patch_idx": 0, "image_id": "A", "band": 0, "row": 10, "col": 20},
        {"patch_idx": 1, "image_id": "A", "band": 0, "row": 11, "col": 21},
        {"patch_idx": 2, "image_id": "B", "band": 1, "row": 12, "col": 22},
        {"patch_idx": 3, "image_id": "B", "band": 1, "row": 13, "col": 23},
    ]
    errors = np.array([0.1, 0.9, 0.2, 0.3])
    z_scores = np.array([0.1, 20.0, 0.2, 0.3])
    records = part_e.aggregate_image_scores(
        rows, errors, z_scores, model_stage=1, analysis_stage=1, baseline_stage=1,
        confounder_labels=["", "vegetation", "", ""])
    a = next(r for r in records if r["image_id"] == "A")
    assert a["raw_condition_score"] == pytest.approx(20.0)
    assert a["condition_score"] == pytest.approx(0.1)
    assert a["n_scored_patches"] == 1
    assert a["n_excluded_patches"] == 1
    assert a["worst_patch_idx"] == 0
    assert a["score_status"] == "confounded"


def test_part_e_aggregate_marks_images_with_too_few_patches_low_support():
    rows = [
        {"patch_idx": 0, "image_id": "A", "band": 0, "row": 10, "col": 20},
        {"patch_idx": 1, "image_id": "B", "band": 0, "row": 11, "col": 21},
        {"patch_idx": 2, "image_id": "B", "band": 1, "row": 12, "col": 22},
        {"patch_idx": 3, "image_id": "B", "band": 1, "row": 13, "col": 23},
        {"patch_idx": 4, "image_id": "B", "band": 2, "row": 14, "col": 24},
    ]
    errors = np.array([0.5, 0.1, 0.2, 0.3, 0.4])
    z_scores = np.array([9.0, 0.1, 0.2, 0.3, 0.4])

    records = part_e.aggregate_image_scores(
        rows, errors, z_scores, model_stage=1, analysis_stage=1, baseline_stage=1)
    by_id = {r["image_id"]: r for r in records}

    assert by_id["A"]["score_status"] == "low_support"
    assert by_id["A"]["n_scored_patches"] == 1
    assert by_id["B"]["score_status"] == "ok"


def test_part_e_confounder_review_uses_excluded_patch():
    records = [
        {"image_id": "A", "worst_patch_idx": 0},
        {"image_id": "B", "worst_patch_idx": 3},
    ]
    rows_by_patch = {
        0: {"image_id": "A", "patch_idx": 0, "band": 0, "row": 10, "col": 20},
        1: {"image_id": "A", "patch_idx": 1, "band": 2, "row": 30, "col": 40},
        2: {"image_id": "A", "patch_idx": 2, "band": 1, "row": 50, "col": 60},
        3: {"image_id": "B", "patch_idx": 3, "band": 0, "row": 70, "col": 80},
    }
    z_by_patch = {0: 9.0, 1: 2.0, 2: 5.0, 3: 1.0}
    label_by_patch = {0: "", 1: "shadow", 2: "marking", 3: ""}

    review = part_e.confounder_review_records(
        records, rows_by_patch, z_by_patch, label_by_patch, n=12)

    assert len(review) == 1
    assert review[0]["image_id"] == "A"
    assert review[0]["worst_patch_idx"] == 2
    assert review[0]["worst_patch_confounder"] == "marking"
    assert review[0]["worst_band"] == 1


def test_part_e_patch_confounder_label_color_rules():
    grass = np.full((128, 128, 3), (40, 120, 40), dtype=np.uint8)
    dry_grass = np.full((128, 128, 3), (60, 115, 145), dtype=np.uint8)
    marking = np.full((128, 128, 3), (220, 220, 220), dtype=np.uint8)
    asphalt = np.full((128, 128, 3), (95, 95, 95), dtype=np.uint8)
    assert patch_confounder_label(grass) == "vegetation"
    assert patch_confounder_label(dry_grass) == "vegetation"
    assert patch_confounder_label(marking) == "marking"
    assert patch_confounder_label(asphalt) == ""


def test_patch_confounder_label_thin_marking_and_shadow():
    asphalt = np.full((128, 128, 3), (95, 95, 95), dtype=np.uint8)
    thin_marking = asphalt.copy()
    thin_marking[:, 58:66] = (210, 210, 210)
    shadow = np.full((128, 128, 3), (135, 135, 135), dtype=np.uint8)
    shadow[:, :58] = (45, 45, 45)
    dark_asphalt = np.full((128, 128, 3), (55, 55, 55), dtype=np.uint8)

    assert patch_confounder_label(thin_marking) == "marking"
    assert patch_confounder_label(shadow) == "shadow"
    assert patch_confounder_label(dark_asphalt) == ""
    assert patch_confounder_label(dark_asphalt, strict=True) == "shadow"


def test_patch_confounder_label_dappled_shadow():
    patch = np.full((128, 128, 3), (125, 125, 125), dtype=np.uint8)
    patch[10:118:2, :] = (60, 60, 60)

    assert patch_confounder_label(patch) == "shadow"


def test_patch_confounder_label_strict_faint_marking():
    asphalt = np.full((128, 128, 3), (95, 95, 95), dtype=np.uint8)
    faint_marking = asphalt.copy()
    faint_marking[:, 62:66] = (140, 140, 140)  # below white v>=145, above strict v>=128

    assert patch_confounder_label(faint_marking) == ""
    assert patch_confounder_label(faint_marking, strict=True) == "marking"


# ---------------------------------------------------------------- stats module

def test_robust_z_basic_and_constant():
    from aibip.stats import robust_z

    z = robust_z([1.0, 2.0, 3.0, 4.0, 100.0])   # median 3, MAD 1
    assert z.shape == (5,)
    assert z[0] < z[1] < 0                       # below median -> negative
    assert z[2] == pytest.approx(0.0)            # == median -> zero
    assert z[3] > 0 and z[3] < 1                 # slightly above median
    assert z[4] > 3.0                            # 100 is a clear robust outlier
    assert robust_z([7.0, 7.0, 7.0]) is None     # MAD == 0 -> no flags possible


def test_outlier_flag_rows_labels_and_order():
    from aibip.stats import outlier_flag_rows

    # m1: median 1.0, MAD 0.1 -> only E (50.0) flags; m2: median 1.1, MAD 0.1
    # -> only D (50.0) flags. Record order (D before E) must beat metric
    # tuple order (m1 first), proving the documented stable ordering.
    records = [{"image_id": "A", "m1": 1.0, "m2": 1.0},
               {"image_id": "B", "m1": 1.1, "m2": 1.0},
               {"image_id": "C", "m1": 0.9, "m2": 1.1},
               {"image_id": "D", "m1": 1.0, "m2": 50.0},
               {"image_id": "E", "m1": 50.0, "m2": 1.2}]
    rows = outlier_flag_rows(records, ("m1", "m2"), stage="X", reason="test_outlier")
    assert [(r["image_id"], r["metric"], r["stage"], r["reason"]) for r in rows] == [
        ("D", "m2", "X", "test_outlier"), ("E", "m1", "X", "test_outlier")]
    assert all(r["value"] == 50.0 and r["z"] > 3.0 for r in rows)
    assert outlier_flag_rows([], ("m1",), stage="X", reason="r") == []
    # a constant metric (MAD == 0) yields no flags even with extreme values
    flat = [{"image_id": i, "m": 1.0} for i in "ABC"] + [{"image_id": "D", "m": 99.0}]
    assert outlier_flag_rows(flat, ("m",), stage="X", reason="r") == []
