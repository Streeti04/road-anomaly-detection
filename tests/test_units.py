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


