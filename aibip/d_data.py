"""Auditable Mini patch extraction, D-only filtering and spatial splitting."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from aibip.b_refactor_compat import compatibility_fingerprint, compatible_code_hashes

from aibip import config, imgio

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/'outputs/d_v2'
DEFAULT_METADATA = ROOT/'data/26-07-13-Cyclomedia-Test-Hammersbach/recordings_bbox_complete.json'


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require_mini(stage):
    if stage != 1:
        raise ValueError('D v2 currently enables Mini / stage 1 only')


def paths(output_root):
    root = Path(output_root)
    return root/'patches/patches_stage1.npy', root/'patches/index_stage1.csv', root/'patches/manifest.json'


def coordinates(metadata):
    result = {}
    for row in read_json(metadata):
        key = row.get('image_id') or row.get('imageId')
        if key and row.get('latitude') is not None and row.get('longitude') is not None:
            pos = [float(row['latitude']), float(row['longitude'])]
            if not np.isfinite(pos).all() or not (-90 <= pos[0] <= 90 and -180 <= pos[1] <= 180):
                raise ValueError(f'Invalid coordinates: {key}')
            if key in result and result[key] != pos:
                raise ValueError(f'Conflicting coordinates for {key}')
            result[key] = pos
    return result


def source_snapshot(metadata):
    """Verify each Mini image and current C PNG/sidecar without invoking SAM."""
    from aibip.c_transfer import _run_key, geometry, compatible_run_keys
    run_key, run_settings = _run_key(config.C_BACKEND)
    accepted_runs = compatible_run_keys(run_key, run_settings)
    coords = coordinates(metadata)
    images = imgio.list_images(ROOT/config.MINI_DIR)
    if len(images) < 2 or len({p.stem for p in images}) != len(images):
        raise ValueError('Mini requires at least two uniquely named images')
    records = []
    for path in images:
        image_id = path.stem
        mask = ROOT/config.MASKS_DIR/f'{image_id}.png'
        sidecar = mask.with_suffix('.test_c.json')
        info = read_json(sidecar)
        image_hash, mask_hash = sha(path), sha(mask)
        if (info.get('source_sha256') != image_hash or info.get('mask_sha256') != mask_hash
                or info.get('cache_key') not in {image_hash+k for k in accepted_runs}):
            raise ValueError(f'C result is stale or changed: {image_id}. Review/regenerate C first.')
        if info.get('status') not in {'ok', 'review', 'no_road'}:
            raise ValueError(f'Invalid C status: {image_id}')
        geo = geometry(info['original_shape'])
        if geo['target_shape'] != info['target_shape'] or geo['target_crop_bounds'] != info['target_crop_bounds']:
            raise ValueError(f'C/B geometry mismatch: {image_id}')
        if image_id not in coords:
            raise ValueError(f'Missing spatial metadata for {image_id}')
        records.append(dict(image_id=image_id, source=str(path.resolve()), source_sha256=image_hash,
                            mask=str(mask.resolve()), mask_sha256=mask_hash,
                            c_sidecar_sha256=sha(sidecar), c_status=info['status'],
                            original_shape=info['original_shape'], work_shape=info['target_shape'],
                            coordinates=coords[image_id]))
    return records


def spatial_split(records, cell_m=300., buffer_m=30., val_fraction=.2, seed=42):
    """Grid groups plus exact file duplicates; buffer removes nearby train images.

    Local equirectangular coordinates are adequate for this small town dataset.
    Neighboring cells can touch, so the optional buffer is checked explicitly.
    """
    if (not np.isfinite([cell_m, buffer_m, val_fraction]).all()
            or cell_m <= 0 or buffer_m < 0 or not 0 < val_fraction < 1):
        raise ValueError('Invalid spatial split settings')
    records = sorted(records, key=lambda r: r['image_id'])
    latlon = np.array([r['coordinates'] for r in records], dtype=float)
    xy = np.column_stack((latlon[:, 1]*111320*np.cos(np.deg2rad(latlon[:, 0].mean())), latlon[:, 0]*111320))
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    by_cell, by_hash = {}, {}
    for i, rec in enumerate(records):
        cell = tuple(np.floor(xy[i]/cell_m).astype(int))
        for mapping, key in [(by_cell, cell), (by_hash, rec['source_sha256'])]:
            if key in mapping:
                union(i, mapping[key])
            else:
                mapping[key] = i
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    group_list = sorted(groups.values(), key=lambda g: records[min(g)]['image_id'])
    if len(group_list) < 2:
        raise ValueError('Only one spatial/duplicate group; choose a smaller --cell-m')
    order = np.random.default_rng(seed).permutation(len(group_list))
    target = max(1, round(len(records)*val_fraction))
    val_idx = []
    for group_index in order[:-1]:  # always reserve at least one train group
        val_idx.extend(group_list[int(group_index)])
        if len(val_idx) >= target:
            break
    val_set = set(val_idx)
    train_idx, buffer_idx = [], []
    for i in range(len(records)):
        if i in val_set:
            continue
        nearest = np.linalg.norm(xy[val_idx]-xy[i], axis=1).min()
        (buffer_idx if nearest < buffer_m else train_idx).append(i)
    if not train_idx:
        raise ValueError('Spatial buffer removed all training images; review split settings')
    ids = lambda indexes: sorted(records[i]['image_id'] for i in indexes)
    return dict(train_ids=ids(train_idx), val_ids=ids(val_idx), buffer_ids=ids(buffer_idx),
                groups=[ids(g) for g in group_list], cell_m=cell_m, buffer_m=buffer_m,
                val_fraction=val_fraction, seed=seed)


def filter_reason(patch, mode):
    if mode == 'none':
        return ''
    if mode == 'legacy':
        from aibip.patch_filter import patch_confounder_label
        label = patch_confounder_label(patch)
        return label if 'vegetation' in label.split('+') else ''
    if mode != 'green':
        raise ValueError('filter must be green, none or legacy')
    h, s, v = cv2.split(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV))
    green = (h >= 35) & (h <= 95) & (s >= 65) & (v >= 40)
    return 'green_vegetation_hint' if green.mean() >= .06 else ''


def spaced_centers(mask, image_id, cap=24, spacing=64, seed=42):
    from aibip.part_d import plan_centers, patch_center_seed
    candidates = plan_centers(mask, image_id, cap=cap*8)
    rng = np.random.default_rng(patch_center_seed(image_id, seed))
    rng.shuffle(candidates)
    selected = []
    # Round-robin bands provide depth diversity; spacing avoids almost identical crops.
    buckets = [[r for r in candidates if r['band'] == band] for band in range(config.N_BANDS)]
    while any(buckets) and len(selected) < cap:
        for bucket in buckets:
            while bucket:
                c = bucket.pop()
                if all((c['row']-p['row'])**2+(c['col']-p['col'])**2 >= spacing**2 for p in selected):
                    selected.append(c)
                    break
            if len(selected) == cap:
                break
    return sorted(selected, key=lambda c: (c['band'], c['row'], c['col']))


def data_code_hashes():
    package = Path(__file__).parent
    # Training-only edits do not invalidate patches.
    return dict({name: sha(package/name) for name in ['d_data.py', 'part_d.py', 'part_b.py', 'patch_filter.py']},
                b_refactor_compat=compatibility_fingerprint())


def validate(output_root=DEFAULT_OUTPUT):
    shard, index, manifest = paths(output_root)
    meta = read_json(manifest)
    if not compatible_code_hashes(meta['code_sha256'], data_code_hashes()):
        raise ValueError('D extraction code changed. Use a new --output-root for extraction.')
    if sha(meta['metadata']) != meta['metadata_sha256'] or source_snapshot(meta['metadata']) != meta['sources']:
        raise ValueError('Images, C masks or coordinates changed since D extraction')
    if sha(shard) != meta['shard_sha256'] or sha(index) != meta['index_sha256']:
        raise ValueError('D patch cache was changed or is incomplete')
    return meta


def extract(stage=1, output_root=DEFAULT_OUTPUT, metadata=DEFAULT_METADATA,
            cap=24, spacing=64, filter_mode='green', appearance='clahe',
            cell_m=300., buffer_m=30., val_fraction=.2, seed=42):
    require_mini(stage)
    if config.PATCH != 128:
        raise ValueError('This D model and its visualizations require 128x128 patches')
    if cap < 1 or spacing < 1 or appearance not in {'clahe', 'original'} or filter_mode not in {'green', 'none', 'legacy'}:
        raise ValueError('Invalid extraction settings')
    root = Path(output_root)
    shard_path, index_path, manifest_path = paths(root)
    settings = dict(cap=cap, spacing=spacing, filter_mode=filter_mode, appearance=appearance,
                    cell_m=cell_m, buffer_m=buffer_m, val_fraction=val_fraction, seed=seed,
                    patch=config.PATCH, bands=config.N_BANDS, width=config.WORK_WIDTH,
                    roi=[config.ROI_TOP_FRAC, config.ROI_BOTTOM_FRAC])
    if manifest_path.exists():
        meta = validate(root)
        if meta['settings'] != settings or str(Path(metadata).resolve()) != meta['metadata']:
            raise ValueError('Extraction settings differ; choose a new --output-root')
        print(f'D: verified existing cache, {meta["n_patches"]} patches')
        return meta
    if shard_path.exists() or index_path.exists():
        raise ValueError('Incomplete/legacy cache found; choose a new --output-root')
    sources = source_snapshot(metadata)
    split = spatial_split(sources, cell_m, buffer_m, val_fraction, seed)
    from aibip.part_b import crop_roi, resize_band, apply_clahe, _label
    from aibip.part_d import write_index
    rows, patches, audit, accepted_examples, rejected_examples = [], [], [], [], []
    r = config.PATCH//2
    for i, src in enumerate(sources, 1):
        mask = cv2.imread(src['mask'], cv2.IMREAD_GRAYSCALE)
        if mask is None or list(mask.shape) != src['work_shape'] or not np.isin(mask, [0, 255]).all():
            raise ValueError(f'Invalid D mask: {src["image_id"]}')
        mask = mask.astype(bool)
        raw = imgio.load_image(src['source'])
        if list(raw.shape) != src['original_shape']:
            raise ValueError(f'Image shape differs from C: {src["image_id"]}')
        proc = resize_band(crop_roi(raw))
        del raw
        if appearance == 'clahe':
            proc = apply_clahe(proc)
        if proc.shape[:2] != mask.shape:
            raise ValueError('Patch/mask coordinates differ')
        planned = spaced_centers(mask, src['image_id'], cap, spacing, seed)
        rejected, accepted = {}, 0
        for center in planned:
            y, x = center['row'], center['col']
            patch = proc[y-r:y+r, x-r:x+r]
            coverage = mask[y-r:y+r, x-r:x+r]
            if patch.shape != (config.PATCH, config.PATCH, 3) or not coverage.all():
                raise ValueError(f'Patch not fully inside road: {src["image_id"]}, {center}')
            reason = filter_reason(patch, filter_mode)
            if reason:
                rejected[reason] = rejected.get(reason, 0)+1
            else:
                rows.append(dict(patch_idx=len(patches), image_id=src['image_id'], **center))
                patches.append(patch.copy())
                accepted += 1
            examples = rejected_examples if reason else accepted_examples
            if len(examples) < 24 and not any(name == src['image_id'] for name, _ in examples):
                tile = np.zeros((config.PATCH+44, config.PATCH, 3), np.uint8)
                tile[44:] = patch
                # small text kept outside the patch; reasons recorded in full in audit JSON
                cv2.putText(tile, src['image_id'], (2, 14), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1)
                cv2.putText(tile, 'rejected' if reason else 'accepted', (2, 32), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1)
                examples.append((src['image_id'], tile))
        audit.append(dict(image_id=src['image_id'], c_status=src['c_status'], candidates=len(planned),
                          accepted=accepted, rejected=rejected,
                          status='ok' if accepted else 'all_filtered' if planned else 'no_geometry',
                          split='val' if src['image_id'] in split['val_ids'] else
                                'train' if src['image_id'] in split['train_ids'] else 'buffer'))
        if i == 1 or i % 10 == 0 or i == len(sources):
            print(f'D extract: {i}/{len(sources)}, accepted {len(patches)}', flush=True)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root/'extraction_audit.json', audit)
    save_json(root/'split_plan.json', split)
    for name, examples in [('accepted', accepted_examples), ('rejected', rejected_examples)]:
        if examples:
            imgio.save_image(root/f'patches_{name}.jpg', imgio.montage([tile for _, tile in examples], cols=6))
    active = {r['image_id'] for r in rows}
    if not active.intersection(split['train_ids']) or not active.intersection(split['val_ids']):
        raise ValueError('No usable train or validation patches; inspect extraction_audit.json and split settings')
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard_path.with_name('patches_stage1.tmp.npy')
    np.save(temporary, np.stack(patches))
    temporary.replace(shard_path)
    write_index(index_path, rows)
    meta = dict(version=2, stage=1, settings=settings, sources=sources, split=split,
                metadata=str(Path(metadata).resolve()), metadata_sha256=sha(metadata),
                code_sha256=data_code_hashes(), index_sha256=sha(index_path), shard_sha256=sha(shard_path),
                n_patches=len(rows), active_images=sorted(active), audit=audit, color_order='BGR')
    save_json(manifest_path, meta)
    print(f'D: {len(rows)} patches, {len(active)} images; spatial train/val/buffer '
          f'{len(active.intersection(split["train_ids"]))}/{len(active.intersection(split["val_ids"]))}/'
          f'{len(active.intersection(split["buffer_ids"]))}. Saved under {root}')
    return meta
