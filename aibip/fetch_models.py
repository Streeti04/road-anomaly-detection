"""Fetch the gitignored model weights into models/.

Both files are pulled from public sources pinned by exact checksum, so any
machine ends up with byte-identical weights:

- mobile_sam_vit_t.pt: official MobileSAM checkpoint (GitHub, renamed)
- segformer-b0-cityscapes.onnx: HF ONNX export, pinned to a specific revision
  (Xenova mirror of nvidia/segformer-b0-finetuned-cityscapes-1024-1024)

Usage:  .venv/bin/python -m aibip.fetch_models
Idempotent: files whose SHA-256 already matches are skipped; a mismatching
file is replaced. Fails loudly if a download's hash does not match.
"""

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

if __package__ in {None, ""}:
    # Allow direct execution via `python aibip/fetch_models.py` in addition to
    # the preferred module form `python -m aibip.fetch_models`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aibip import config  # torch-free constants module: single source for MODELS_DIR

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / config.MODELS_DIR

# name -> (url, sha256, size_bytes)
SOURCES = {
    "mobile_sam_vit_t.pt": (
        "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
        40_728_226,
    ),
    "segformer-b0-cityscapes.onnx": (
        "https://huggingface.co/Xenova/segformer-b0-finetuned-cityscapes-1024-1024"
        "/resolve/64e537ca2bf6bf2afc13727ee6bb61cfca9e1048/onnx/model.onnx",
        "142427caa45a47d57d5bd23b62c3bfc852fa7df0c6c2291c5ef34f581fa406bf",
        15_200_774,
    ),
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str, expected_sha: str, size: int) -> str:
    tmp = MODELS_DIR / f"{name}.part"
    print(f"downloading {name} ({size / 1e6:.1f} MB) ...")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    actual = sha256_of(tmp)
    if actual != expected_sha:
        tmp.unlink()
        raise SystemExit(
            f"checksum mismatch for {name}:\n  expected {expected_sha}\n  got      {actual}"
        )
    out = MODELS_DIR / name
    tmp.replace(out)
    return f"downloaded  {name}  sha256 ok"


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name, (url, sha, size) in SOURCES.items():
        path = MODELS_DIR / name
        if path.exists():
            actual = sha256_of(path)
            if actual == sha:
                results.append(f"ok          {name}  (already present)")
                continue
            print(f"{name}: hash mismatch, re-downloading")
        results.append(fetch(name, url, sha, size))
    print("\n".join(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
