"""Test C pipeline, adapted to this project's raw-BGR -> B-work-mask interface.

B prepares C's unenhanced 45--73% RGB view and D's historical work view.
Masks outside the Test crop are zero, never extrapolated onto the hood.
"""
import csv
import hashlib
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from aibip import config, imgio, pool_data, part_b
from aibip.b_refactor_compat import compatibility_fingerprint, compatible_run_keys
from aibip.c_test_core import TestRoadSegmenter

C_TOP = part_b.SEGMENT_TOP
C_BOTTOM = part_b.SEGMENT_BOTTOM
C_WIDTH = part_b.SEGMENT_WIDTH
VERSION = 'test-road-c-1'


def _api():
    from aibip import part_c
    return part_c


def _hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def geometry(raw_shape):
    """Compatibility name; B owns the coordinate mapping."""
    return part_b.segmentation_geometry(raw_shape)


def crop_rgb(raw_bgr):
    return part_b.segmentation_crop_rgb(raw_bgr)


def project_mask(small_mask, raw_shape):
    return part_b.mask_to_work_view(small_mask, raw_shape)


def segment_details(image, backend=None):
    """Run unchanged Test scoring, returning target-aligned bool mask and audit data."""
    import torch
    torch.set_num_threads(4)
    prepared, geo = part_b.preprocess_for_segmentation(image)
    predictor = _api()._get_predictor(backend)
    model = TestRoadSegmenter(predictor, C_WIDTH)
    # predict returns a native crop mask too; the reduced mask is authoritative
    # for the adapter, exactly as in Test's saved NPZ artifacts.
    _, y0, width, y1 = geo['crop_bounds']
    full, small_mask, small_rgb, info = model.predict_preprocessed(prepared, (y1-y0, width))
    del full, prepared
    info.update(geo)
    info['inference_shape'] = list(small_mask.shape)
    info['original_shape'] = list(image.shape)
    return project_mask(small_mask, image.shape), small_mask, small_rgb, info


def segment_road(image, backend=None):
    return segment_details(image, backend)[0]


def _run_key(backend):
    backend = backend or config.C_BACKEND
    if backend not in {'mobile_sam', 'sam'}:
        raise ValueError(f'Unknown backend: {backend}')
    weights = _api().ROOT / config.MODELS_DIR / (
        'mobile_sam_vit_t.pt' if backend == 'mobile_sam' else 'sam_vit_b_01ec64.pth')
    if not weights.is_file():
        raise FileNotFoundError(f'Missing model weights: {weights}')
    settings = dict(version=VERSION, backend=backend, weights_sha256=_hash(weights),
                    c_crop=[C_TOP, C_BOTTOM], c_width=C_WIDTH,
                    b_crop=[config.ROI_TOP_FRAC, config.ROI_BOTTOM_FRAC],
                    b_width=config.WORK_WIDTH, b_clahe=config.USE_CLAHE)
    package = Path(__file__).parent
    settings['code_sha256'] = {name: _hash(package/name) for name in
                               ['c_test_core.py', 'c_transfer.py', 'part_c.py', 'part_b.py']}
    settings['code_sha256']['b_refactor_compat'] = compatibility_fingerprint()
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(), settings


def _paths(stem):
    return (config.MASKS_DIR/f'{stem}.png',
            config.MASKS_DIR/f'{stem}.test_c.json',
            config.OUTPUTS_DIR/'qc'/'test_c'/f'{stem}.jpg')


def _cached(mask, meta, preview, key, compatible_keys=()):
    if not (mask.is_file() and meta.is_file() and preview.is_file()):
        return None
    try:
        record = json.loads(meta.read_text(encoding='utf-8'))
        if record['cache_key'] in {key, *compatible_keys} and record['mask_sha256'] == _hash(mask):
            decoded = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            if decoded is not None and list(decoded.shape) == record['target_shape']:
                return record
    except (ValueError, KeyError, OSError):
        pass
    return None


def _backup(files):
    existing = [p for p in files if p.exists()]
    if existing:
        dest = config.OUTPUTS_DIR/'c_before_test_transfer'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        dest.mkdir(parents=True)
        for path in existing:
            shutil.copy2(path, dest/path.name)


def _preview(small_rgb, small_mask, info, stem):
    image = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)
    image[small_mask] = (.6*image[small_mask] + .4*np.array([90, 230, 0])).astype(np.uint8)
    for (x, y), label in zip(info['prompts'], info['labels']):
        cv2.circle(image, (int(x), int(y)), 4, (255, 100, 50) if label else (255, 0, 255), -1)
    out = np.zeros((image.shape[0]+34, image.shape[1], 3), np.uint8)
    out[34:] = image
    _api()._label(out, (8, 23), f"{stem} | {info['status']} | C crop 45-73%")
    return out


def _run_paths(paths, backend=None, force=False):
    if not paths:
        raise ValueError('No images selected')
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('Image IDs must be unique')
    run_key, settings = _run_key(backend)
    accepted_runs = compatible_run_keys(run_key, settings)
    # Preflight all selected files before modifying any mask. Old caches require
    # explicit --force; source/config/weights changes cannot be silently skipped.
    jobs = []
    print(f'Preflight: hashing {len(paths)} images for cache check ...', flush=True)
    for i, path in enumerate(paths, 1):
        mask, meta, preview = _paths(path.stem)
        source_hash = _hash(path)
        key = source_hash + run_key
        cached = _cached(mask, meta, preview, key, {source_hash+k for k in accepted_runs})
        if mask.exists() and cached is None and not force:
            raise ValueError(f'Existing incompatible mask {mask}. Use --force to back it up and regenerate.')
        if i % 500 == 0 or i == len(paths):
            print(f'Preflight: {i}/{len(paths)}', flush=True)
        jobs.append((path, mask, meta, preview, key, source_hash, cached))
    records, errors, reused = [], [], 0
    start = time.perf_counter()
    for i, (path, mask_path, meta, preview, key, source_hash, cached) in enumerate(jobs, 1):
        try:
            if cached is not None and not force:
                record = cached
                reused += 1
            else:
                mask, small_mask, small_rgb, info = segment_details(imgio.load_image(path), backend)
                stats = _api().mask_stats(mask, info.get('confidence', 0.0))
                _backup([mask_path, meta, preview])
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                preview.parent.mkdir(parents=True, exist_ok=True)
                temporary = mask_path.with_name(mask_path.stem+'.tmp.png')
                imgio.save_image(temporary, mask.astype(np.uint8)*255)
                temporary.replace(mask_path)
                imgio.save_image(preview, _preview(small_rgb, small_mask, info, path.stem))
                record = dict(image_id=path.stem, source=str(path.resolve()), source_sha256=source_hash,
                              cache_key=key, mask_sha256=_hash(mask_path), stats=stats, **info)
                _save_json(meta, record)
            records.append(record)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(dict(image_id=path.stem, status='error', error=str(exc)))
            print(f'C ERROR {path.name}: {exc}', flush=True)
        if i == 1 or i % 5 == 0 or i == len(jobs):
            print(f'C Test: {i}/{len(jobs)}, {time.perf_counter()-start:.1f}s, cached={reused}', flush=True)
    _save_json(config.OUTPUTS_DIR/'c_test_manifest.json',
               dict(settings=settings, records=records, errors=errors))
    # Keep statistics for unrelated images, e.g. when single() updates one mask.
    stats_path = config.OUTPUTS_DIR/'mask_stats.csv'
    existing = {}
    if stats_path.exists():
        with stats_path.open(newline='', encoding='utf-8') as file:
            existing = {r['image_id']: r for r in csv.DictReader(file)}
    for record in records:
        existing[record['image_id']] = dict(image_id=record['image_id'], **record['stats'])
    for error in errors:
        existing.pop(error['image_id'], None)
    with stats_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=_api().STATS_FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(existing.values())
    selected_ids = {p.stem for p in paths}
    flags = []
    if config.FLAGS_CSV.exists():
        with config.FLAGS_CSV.open(newline='', encoding='utf-8') as file:
            flags = [r for r in csv.DictReader(file)
                     if r['stage'] == 'C' and r['image_id'] not in selected_ids]
    for record in records + errors:
        if record['status'] != 'ok':
            flags.append(dict(image_id=record['image_id'], stage='C',
                              reason='test_c_'+record['status'], metric='candidate_score',
                              value=record.get('score', ''), z=''))
    _api().write_flags(flags, stage='C')
    # Per-image statuses and all candidate scores are authoritative in sidecars.
    print(f'C: {len(records)} successful, {len(errors)} errors; mask/status/overlay files saved.')
    pool_patches = pool_data.paths(pool_data.DEFAULT_OUTPUT)[0]
    if pool_patches.exists():
        print('D patch caches already exist under outputs/pool/patches: they still refer to '
              'the previous masks. Do not treat them as patches extracted from this C run.')
    if errors:
        raise RuntimeError('C finished with image errors; inspect c_test_manifest.json before running D')
    return records


def run(data_dir, backend=None, force=False, limit=None):
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    return _run_paths(imgio.list_images(data_dir)[:limit], backend, force)


def single(image_path, backend=None, force=False):
    return _run_paths([Path(image_path)], backend, force)


def qc(data_dir, n=12, seed=config.SEED, backend=None):
    """Montage existing validated overlays only; no extra SAM inference."""
    if n < 1:
        raise ValueError('n must be positive')
    paths = imgio.list_images(data_dir)
    if not paths:
        raise ValueError('No images found')
    selected = random.Random(seed).sample(paths, min(n, len(paths)))
    key, settings = _run_key(backend)
    accepted_runs = compatible_run_keys(key, settings)
    tiles = []
    for path in selected:
        mask, meta, preview = _paths(path.stem)
        source_hash = _hash(path)
        if _cached(mask, meta, preview, source_hash+key, {source_hash+k for k in accepted_runs}) is None:
            raise ValueError(f'Missing/current C result required for {path.name}; run C first')
        tiles.append(imgio.load_image(preview))
    dest = config.OUTPUTS_DIR/'qc'/'c_test_overlays.jpg'
    dest.parent.mkdir(parents=True, exist_ok=True)
    imgio.save_image(dest, imgio.montage(tiles, cols=3))
    print(f'QC: {dest} ({len(tiles)} saved overlays; no inference)')


def benchmark(data_dir, n=4, backend=None):
    """Time actual crop, five prompts, selection and projection; write no masks."""
    if n < 1:
        raise ValueError('n must be positive')
    paths = imgio.list_images(data_dir)[:n]
    if not paths:
        raise ValueError('No images found')
    segment_road(imgio.load_image(paths[0]), backend)
    device = _api()._torch_device()
    durations = []
    for path in paths:
        image = imgio.load_image(path)
        _api()._sync_device(device)
        start = time.perf_counter()
        segment_road(image, backend)
        _api()._sync_device(device)
        durations.append(time.perf_counter()-start)
    print(f'C Test benchmark: {np.mean(durations):.3f}s/image ({len(paths)} images, '
          'including crop/selection/projection, excluding file IO and warmup)')
