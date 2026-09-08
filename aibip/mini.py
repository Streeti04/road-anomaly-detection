"""Deterministic mini-set builder.

The official mini set is a fixed random sample of 100 source images, drawn
once with a seeded RNG so every run and every machine selects the SAME files.
The set is materialized as symlinks into data/mini when the OS permits it. On
Windows without Developer Mode/Admin rights, file symlink creation can fail
with WinError 1314; in that case the selected files are copied instead so the
project remains runnable.
"""

import random
import shutil
from pathlib import Path

from aibip import config, imgio


def _materialize(src: Path, dst: Path) -> None:
    """Create a symlink when possible, otherwise copy the file.

    Copy fallback is mainly for Windows machines where ordinary users often do
    not have the SeCreateSymbolicLinkPrivilege permission.
    """
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def make_mini(n=config.N_MINI, seed=config.SEED, source=None, dest=None) -> list[Path]:
    """Build the official 100-image mini set.

    Deterministic: `n` filenames are sampled with random.Random(seed) from the
    SORTED source listing, so the same seed always yields the same set. Each
    chosen file gets materialized as `dest/<name>`: symlink where supported,
    copy fallback where symlink creation is denied. Existing files are skipped
    (idempotent). Returns the sorted list of materialized mini-image
    paths.
    """
    files = imgio.list_images(source or config.MINI_SOURCE_DIR)
    if len(files) < n:
        raise ValueError(f"need {n} images, found only {len(files)} in {source or config.MINI_SOURCE_DIR}")
    chosen = sorted(random.Random(seed).sample(files, n))
    dest = Path(dest or config.MINI_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    links = []
    for src in chosen:
        link = dest / src.name
        if not (link.is_symlink() or link.exists()):
            _materialize(src, link)
        links.append(link)
    return sorted(links)


if __name__ == "__main__":
    make_mini()
    print(len(list(config.MINI_DIR.iterdir())))
