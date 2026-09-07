"""Small, pure image I/O helpers.

cv2 loads images as BGR uint8 arrays (shape HxWx3) and writes them back the
same way; all functions here preserve that convention.
"""

import math
from pathlib import Path

import cv2
import numpy as np

from aibip import config


def list_images(directory) -> list:
    """Return a plain sorted() list of image file paths directly under `directory`.

    Sorting is the plain string sort on Path objects (case-sensitive, uppercase
    before lowercase). Non-recursive: files in subdirectories are ignored.
    """
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMG_EXTENSIONS
    )


def load_image(path) -> np.ndarray:
    """Load an image as BGR uint8 (HxWx3) via cv2.imread.

    Raises FileNotFoundError if cv2 returns None (missing/undecodable file) —
    cv2 signals failure with None, not an exception.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cv2 could not read image: {path}")
    return image


def resize(image: np.ndarray, res=config.WORK_RES) -> np.ndarray:
    """Resize `image` to res=(height, width) with INTER_AREA (correct for downscaling)."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("resize() needs a non-empty ndarray")
    height, width = res
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def to_work_resolution(image: np.ndarray) -> np.ndarray:
    """Convenience wrapper: resize() to config.WORK_RES."""
    return resize(image, config.WORK_RES)


def montage(images: list, cols: int = 4, pad: int = 4) -> np.ndarray:
    """Tile same-channel uint8 images into a grid, ready for cv2.imwrite.

    Layout convention: every tile gets a black border of `pad` pixels on all
    sides; images of different sizes are resized to the size of the first
    image; an incomplete last row is filled with black tiles so the grid is
    always fully filled with rows = ceil(n / cols).
    Output shape: (rows*(h + 2*pad), cols*(w + 2*pad) [, channels]).
    """
    if not images:
        raise ValueError("montage() needs at least one image")
    height, width = images[0].shape[:2]
    tiles = [resize(im, (height, width)) for im in images]
    rows = math.ceil(len(tiles) / cols)
    channels = () if tiles[0].ndim == 2 else (tiles[0].shape[2],)
    canvas = np.zeros((rows * (height + 2 * pad), cols * (width + 2 * pad), *channels), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        row, col = divmod(i, cols)
        y = row * (height + 2 * pad) + pad
        x = col * (width + 2 * pad) + pad
        canvas[y:y + height, x:x + width] = tile
    return canvas


def save_image(path, image: np.ndarray) -> None:
    """Write `image` (BGR uint8) via cv2.imwrite; raise IOError on failure.

    cv2.imwrite returns False silently on failure (bad path, unsupported
    extension), so the result is checked explicitly.
    """
    if not cv2.imwrite(str(path), image):
        raise IOError(f"cv2.imwrite failed for: {path}")
