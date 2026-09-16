"""Stage-2 data adapter; preserves the existing Mini extraction and its hashes."""
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

from aibip.b_refactor_compat import compatible_code_hashes, compatible_plan

from aibip import config, imgio, d_data as mini
from aibip.d_data import sha, read_json, save_json

ROOT = mini.ROOT
STAGE = 2
DEFAULT_POOL = ROOT/'data/26-07-13-Cyclomedia-Test-Hammersbach'
DEFAULT_OUTPUT = ROOT/'outputs/d_stage2'


def safe_root(output_root):
    root = Path(output_root).resolve()
    protected = mini.DEFAULT_OUTPUT.resolve()
    if root == protected or root in protected.parents or protected in root.parents:
        raise ValueError('Stage 2 needs a separate output root outside outputs/d_v2')
    return root


def paths(output_root):
    root = safe_root(output_root)
    return root/'patches/patches_stage2.npy', root/'patches/index_stage2.csv', root/'patches/manifest.json'


def inventory(pool_dir=DEFAULT_POOL, metadata=mini.DEFAULT_METADATA, mini_dir=None):
    """List actual files; metadata entries without images are explicitly reported."""
    pool_dir = Path(pool_dir).resolve()
    mini_dir = Path(mini_dir or ROOT/config.MINI_DIR).resolve()
    images, mini_images = imgio.list_images(pool_dir), imgio.list_images(mini_dir)
    ids, mini_ids = [p.stem for p in images], [p.stem for p in mini_images]
    if not mini_images or len(images) <= len(mini_images):
        raise ValueError('Stage 2 requires a pool larger than the nonempty Mini set')
    if len(set(ids)) != len(ids) or len(set(mini_ids)) != len(mini_ids):
        raise ValueError('Image IDs must be unique within each directory')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', key) for key in ids):
        raise ValueError('Image IDs must contain only letters, digits, underscores or hyphens')
    coords = mini.coordinates(metadata)
    missing_coordinates = sorted(set(ids)-coords.keys())
    missing_mini = sorted(set(mini_ids)-set(ids))
    if missing_coordinates or missing_mini:
        raise ValueError(f'Missing coordinates: {missing_coordinates[:10]}; Mini absent from pool: {missing_mini[:10]}')
    pool_by_id = {p.stem: p for p in images}
    # An ID match alone is insufficient for reuse and the later Mini evaluation.
    different = [p.stem for p in mini_images if sha(p) != sha(pool_by_id[p.stem])]
    if different:
        raise ValueError(f'Mini and pool have different bytes for the same IDs: {different[:10]}')
    metadata_ids = {r.get('image_id') or r.get('imageId') for r in read_json(metadata)}-{None, ''}
    return dict(pool_dir=str(pool_dir), metadata=str(Path(metadata).resolve()),
                mini_dir=str(mini_dir), n_images=len(images), n_mini=len(mini_images),
                metadata_unique_ids=len(metadata_ids), image_ids=ids, mini_ids=sorted(mini_ids),
                missing_image_ids=sorted(metadata_ids-set(ids)),
                raw_bytes=sum(p.stat().st_size for p in images),
                note='Actual local pool; not a claim of 10000 training images')


def source_snapshot(pool_dir, metadata):
    """Validate the unchanged accepted C pipeline's PNGs and coordinate mapping."""
    from aibip.c_transfer import _run_key, geometry, compatible_run_keys
    run_key, run_settings = _run_key(config.C_BACKEND)
    accepted_runs = compatible_run_keys(run_key, run_settings)
    coords = mini.coordinates(metadata)
    sources = []
    for i, path in enumerate(imgio.list_images(pool_dir), 1):
        mask = ROOT/config.MASKS_DIR/f'{path.stem}.png'
        sidecar = mask.with_suffix('.test_c.json')
        if not mask.exists() or not sidecar.exists():
            raise ValueError(f'Missing C result: {path.stem}; run the stage2 segment command first')
        info = read_json(sidecar)
        image_hash, mask_hash = sha(path), sha(mask)
        if (info.get('source_sha256') != image_hash or info.get('mask_sha256') != mask_hash
                or info.get('cache_key') not in {image_hash+k for k in accepted_runs}):
            raise ValueError(f'Stale C result: {path.stem}; review/regenerate C before extraction')
        if info.get('status') not in {'ok', 'review', 'no_road'}:
            raise ValueError(f'Invalid C status: {path.stem}')
        geo = geometry(info['original_shape'])
        if geo['target_shape'] != info['target_shape'] or geo['target_crop_bounds'] != info['target_crop_bounds']:
            raise ValueError(f'C/B coordinates differ: {path.stem}')
        sources.append(dict(image_id=path.stem, source=str(path.resolve()), source_sha256=image_hash,
                            mask=str(mask.resolve()), mask_sha256=mask_hash, c_sidecar_sha256=sha(sidecar),
                            c_status=info['status'], original_shape=info['original_shape'],
                            work_shape=info['target_shape'], coordinates=coords[path.stem]))
        if i % 100 == 0:
            print(f'D source verification: {i} images', flush=True)
    return sources


def code_hashes():
    return dict(mini.data_code_hashes(), **{'d_pool.py': sha(__file__),
                'imgio.py': sha(Path(__file__).with_name('imgio.py')),
                'config.py': sha(Path(__file__).with_name('config.py'))})


def validate(output_root=DEFAULT_OUTPUT):
    shard, index, manifest = paths(output_root)
    meta = read_json(manifest)
    if meta.get('stage') != 2 or not compatible_code_hashes(meta['code_sha256'], code_hashes()):
        raise ValueError('Stage-2 extraction code changed; choose a new --output-root')
    if sha(meta['metadata']) != meta['metadata_sha256']:
        raise ValueError('Stage-2 metadata changed')
    if source_snapshot(meta['pool_dir'], meta['metadata']) != meta['sources']:
        raise ValueError('Stage-2 source images or C results changed')
    if sha(shard) != meta['shard_sha256'] or sha(index) != meta['index_sha256']:
        raise ValueError('Stage-2 patch cache changed or incomplete')
    return meta


def extract_image(src, settings):
    """Same preprocessing, sampling and full-mask coverage as the validated Mini run."""
    from aibip.part_b import crop_roi, resize_band, apply_clahe
    mask = cv2.imread(src['mask'], cv2.IMREAD_GRAYSCALE)
    if mask is None or list(mask.shape) != src['work_shape'] or not np.isin(mask, [0, 255]).all():
        raise ValueError(f'Invalid road mask: {src["image_id"]}')
    mask = mask.astype(bool)
    raw = imgio.load_image(src['source'])
    if list(raw.shape) != src['original_shape']:
        raise ValueError(f'Image shape changed: {src["image_id"]}')
    proc = resize_band(crop_roi(raw))
    del raw
    if settings['appearance'] == 'clahe':
        proc = apply_clahe(proc)
    if proc.shape[:2] != mask.shape:
        raise ValueError('Patch and mask coordinates differ')
    planned = mini.spaced_centers(mask, src['image_id'], settings['cap'], settings['spacing'], settings['seed'])
    patches, centers, rejected = [], [], {}
    radius = config.PATCH//2
    for center in planned:
        y, x = center['row'], center['col']
        patch = proc[y-radius:y+radius, x-radius:x+radius]
        if patch.shape != (128, 128, 3) or not mask[y-radius:y+radius, x-radius:x+radius].all():
            raise ValueError(f'Patch outside road: {src["image_id"]}')
        reason = mini.filter_reason(patch, settings['filter_mode'])
        if reason:
            rejected[reason] = rejected.get(reason, 0)+1
        else:
            patches.append(patch.copy())
            centers.append(center)
    array = np.stack(patches) if patches else np.empty((0, 128, 128, 3), np.uint8)
    audit = dict(image_id=src['image_id'], c_status=src['c_status'], candidates=len(planned),
                 accepted=len(patches), rejected=rejected,
                 status='ok' if patches else 'all_filtered' if planned else 'no_geometry')
    return array, centers, audit


def extract(pool_dir=DEFAULT_POOL, metadata=mini.DEFAULT_METADATA, output_root=DEFAULT_OUTPUT,
            mini_reference=mini.DEFAULT_OUTPUT):
    root = safe_root(output_root)
    shard, index, manifest = paths(root)
    mini_meta = mini.validate(mini_reference)
    settings = mini_meta['settings']
    if settings['patch'] != 128:
        raise ValueError('This model requires 128x128 patches')
    if (config.PATCH != settings['patch'] or config.N_BANDS != settings['bands']
            or config.WORK_WIDTH != settings['width']
            or [config.ROI_TOP_FRAC, config.ROI_BOTTOM_FRAC] != settings['roi']):
        raise ValueError('Current geometry differs from the Mini reference settings')
    if manifest.exists():
        meta = validate(root)
        if (meta['pool_dir'] != str(Path(pool_dir).resolve()) or meta['metadata'] != str(Path(metadata).resolve())
                or meta['mini_manifest_sha256'] != sha(mini.paths(mini_reference)[2])):
            raise ValueError('Different extraction inputs; use a new --output-root')
        print(f'D Stage 2: verified {meta["n_patches"]} cached patches')
        return meta
    info = inventory(pool_dir, metadata)
    sources = source_snapshot(pool_dir, metadata)
    split = mini.spatial_split(sources, settings['cell_m'], settings['buffer_m'],
                               settings['val_fraction'], settings['seed'])
    plan = dict(stage=2, pool_dir=info['pool_dir'], metadata=info['metadata'],
                metadata_sha256=sha(metadata), sources=sources, settings=settings, split=split,
                inventory=info, code_sha256=code_hashes(),
                mini_manifest_sha256=sha(mini.paths(mini_reference)[2]))
    plan_path = root/'extraction_plan.json'
    if plan_path.exists() and not compatible_plan(read_json(plan_path), plan):
        raise ValueError('Partial extraction has different inputs/code; use a new --output-root')
    if not plan_path.exists() and (shard.exists() or index.exists() or (root/'chunks').exists()):
        raise ValueError('Untracked partial cache; use a new --output-root')
    root.mkdir(parents=True, exist_ok=True)
    # Chunk files plus final uint8 memmap, with a reserve. Conservative on resume.
    required = len(sources)*settings['cap']*128*128*3*2 + 512*1024**2
    if shutil.disk_usage(root).free < required:
        raise ValueError(f'Insufficient free space; reserve at least {required/1024**3:.1f} GiB')
    save_json(plan_path, plan)
    save_json(root/'inventory.json', info)
    save_json(root/'split_plan.json', split)
    chunk_dir = root/'chunks'
    chunk_dir.mkdir(exist_ok=True)
    rows, audit, examples = [], [], []
    membership = {key: set(split[key+'_ids']) for key in ['train', 'val', 'buffer']}
    for i, src in enumerate(sources, 1):
        chunk = chunk_dir/f'{src["image_id"]}.npy'
        sidecar = chunk.with_suffix('.json')
        if sidecar.exists():
            saved = read_json(sidecar)
            if not chunk.exists() or sha(chunk) != saved['sha256']:
                raise ValueError(f'Changed chunk: {chunk}; choose a new output root')
        else:
            array, centers, record = extract_image(src, settings)
            temporary = chunk.with_suffix('.tmp.npy')
            np.save(temporary, array)
            temporary.replace(chunk)
            saved = dict(centers=centers, audit=record, sha256=sha(chunk))
            save_json(sidecar, saved)  # completion marker written last
        for center in saved['centers']:
            rows.append(dict(patch_idx=len(rows), image_id=src['image_id'], **center))
        record = dict(saved['audit'], split=next(k for k, ids in membership.items() if src['image_id'] in ids))
        audit.append(record)
        if record['accepted'] and len(examples) < 24:
            sample = np.load(chunk, mmap_mode='r')
            tile = np.zeros((160, 128, 3), np.uint8)
            tile[32:] = sample[0]
            cv2.putText(tile, src['image_id'], (2, 14), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1)
            cv2.putText(tile, record['split'], (2, 28), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1)
            examples.append(tile)
            del sample
        if i == 1 or i % 50 == 0 or i == len(sources):
            print(f'D Stage 2 extraction: {i}/{len(sources)}, {len(rows)} patches (completed chunks reused)', flush=True)
    save_json(root/'extraction_audit.json', audit)
    if examples:
        imgio.save_image(root/'patches_accepted.jpg', imgio.montage(examples, cols=6))
    active = {r['image_id'] for r in rows}
    if not active.intersection(split['train_ids']) or not active.intersection(split['val_ids']):
        raise ValueError('No usable train or validation patches; inspect extraction_audit.json')
    shard.parent.mkdir(exist_ok=True)
    temporary = shard.with_suffix('.tmp.npy')
    output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8, shape=(len(rows),128,128,3))
    offset = 0
    for src in sources:
        chunk = np.load(chunk_dir/f'{src["image_id"]}.npy', mmap_mode='r')
        output[offset:offset+len(chunk)] = chunk
        offset += len(chunk)
        del chunk
    output.flush()
    del output  # close the mapping before rename on Windows
    temporary.replace(shard)
    from aibip.part_d import write_index
    temporary_index = index.with_suffix('.tmp.csv')
    write_index(temporary_index, rows)
    temporary_index.replace(index)
    meta = dict(plan, version=3, shard_sha256=sha(shard), index_sha256=sha(index),
                n_patches=len(rows), active_images=sorted(active), audit=audit, color_order='BGR')
    save_json(manifest, meta)
    print(f'D Stage 2 ready: {len(rows)} patches from {len(active)}/{len(sources)} images; {root}')
    return meta
