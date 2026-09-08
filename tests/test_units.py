"""Pure-function unit tests. No models, no data directory, no GPU."""

import csv
from pathlib import Path

import numpy as np
import pytest

from aibip import config, imgio, mini, part_b
from aibip.part_b import preprocess


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


