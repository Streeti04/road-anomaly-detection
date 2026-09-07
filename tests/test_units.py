"""Pure-function unit tests. No models, no data directory, no GPU."""

import csv
from pathlib import Path

import numpy as np
import pytest

from aibip import config, imgio


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
