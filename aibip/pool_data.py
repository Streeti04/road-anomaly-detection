"""Local pool input without any Mini data or previously generated artifacts.

Uses the same road-patch sampler and appearance as the validated Stage-2 run.
Historical d_pool.py is retained unchanged so existing checkpoints stay usable.
"""
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

from aibip.b_refactor_compat import compatible_code_hashes, compatible_plan

from aibip import config, imgio, d_data as common
from aibip.d_data import sha, read_json, save_json
from aibip.d_pool import extract_image

ROOT = common.ROOT
STAGE = 2
DEFAULT_OUTPUT = ROOT/'outputs/pool'


def safe_root(output_root):
    root = Path(output_root).resolve()
    for protected in [common.DEFAULT_OUTPUT.resolve(), (ROOT/'outputs/d_stage2').resolve()]:
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError('Use a separate output root outside the historical D results')
    return root


def paths(output_root):
    root = safe_root(output_root)
    return root/'patches/patches_stage2.npy', root/'patches/index_stage2.csv', root/'patches/manifest.json'


def find_images(directory):
    """Recursive discovery, deterministic ordering; duplicate stems are ambiguous."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f'Image directory does not exist: {root}')
    images = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in config.IMG_EXTENSIONS)
    ids = [p.stem for p in images]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate image IDs in image directory/subdirectories; select one image variant per ID')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', key) for key in ids):
        raise ValueError('Image IDs must contain only letters, digits, underscores or hyphens')
    return images


def inventory(pool_dir, metadata):
    """Only local images and their recording metadata are required. No Mini IO."""
    pool_dir = Path(pool_dir).resolve()
    images = find_images(pool_dir)
    ids = [p.stem for p in images]
    if len(images) < 2:
        raise ValueError('At least two pool images are required')
    coords = common.coordinates(metadata)
    missing_coordinates = sorted(set(ids)-coords.keys())
    if missing_coordinates:
        raise ValueError(f'Missing coordinates for {len(missing_coordinates)} images: {missing_coordinates[:10]}')
    metadata_ids = {r.get('image_id') or r.get('imageId') for r in read_json(metadata)}-{None, ''}
    return dict(pool_dir=str(pool_dir), metadata=str(Path(metadata).resolve()),
                n_images=len(images), metadata_unique_ids=len(metadata_ids), image_ids=ids,
                missing_image_ids=sorted(metadata_ids-set(ids)),
                raw_bytes=sum(p.stat().st_size for p in images),
                note='Actual local files, recursively discovered; no Mini dependency')


def source_snapshot(pool_dir, metadata):
    """Validate the unchanged accepted C pipeline's PNGs and coordinate mapping."""
    from aibip.c_transfer import _run_key, geometry, compatible_run_keys
    run_key, run_settings = _run_key(config.C_BACKEND)
    accepted_runs = compatible_run_keys(run_key, run_settings)
    coords = common.coordinates(metadata)
    sources = []
    for i, path in enumerate(find_images(pool_dir), 1):
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
    return dict(common.data_code_hashes(), **{'pool_data.py': sha(__file__),
                'd_pool.py': sha(Path(__file__).with_name('d_pool.py')),
                'imgio.py': sha(Path(__file__).with_name('imgio.py')),
                'config.py': sha(Path(__file__).with_name('config.py'))})


def validate(output_root=DEFAULT_OUTPUT):
    shard, index, manifest = paths(output_root)
    meta = read_json(manifest)
    if meta.get('stage') != 2 or meta.get('adapter') != 'portable_pool' or not compatible_code_hashes(meta['code_sha256'], code_hashes()):
        raise ValueError('Stage-2 extraction code changed; choose a new --output-root')
    if sha(meta['metadata']) != meta['metadata_sha256']:
        raise ValueError('Stage-2 metadata changed')
    if source_snapshot(meta['pool_dir'], meta['metadata']) != meta['sources']:
        raise ValueError('Stage-2 source images or C results changed')
    if sha(shard) != meta['shard_sha256'] or sha(index) != meta['index_sha256']:
        raise ValueError('Stage-2 patch cache changed or incomplete')
    return meta



def extraction_settings(cap=24, spacing=64, appearance='clahe', filter_mode='green',
                        cell_m=300., buffer_m=30., val_fraction=.2, seed=42):
    if cap < 1 or spacing < 1 or appearance not in {'clahe','original'} or filter_mode not in {'green','none','legacy'}:
        raise ValueError('Invalid patch extraction settings')
    if not np.isfinite([cell_m,buffer_m,val_fraction]).all() or cell_m <= 0 or buffer_m < 0 or not 0 < val_fraction < 1:
        raise ValueError('Invalid spatial split settings')
    if config.PATCH != 128:
        raise ValueError('This model requires 128x128 patches')
    return dict(cap=cap, spacing=spacing, appearance=appearance, filter_mode=filter_mode,
                cell_m=cell_m, buffer_m=buffer_m, val_fraction=val_fraction, seed=seed,
                patch=config.PATCH, bands=config.N_BANDS, width=config.WORK_WIDTH,
                roi=[config.ROI_TOP_FRAC,config.ROI_BOTTOM_FRAC])


def extract(pool_dir, metadata, output_root=DEFAULT_OUTPUT, **options):
    root = safe_root(output_root)
    pool_path = Path(pool_dir).resolve()
    if root == pool_path or pool_path in root.parents:
        raise ValueError('Output directory must be outside the recursively scanned image directory')
    shard, index, manifest = paths(root)
    settings = extraction_settings(**options)
    if manifest.exists():
        meta = validate(root)
        if (meta['pool_dir'] != str(Path(pool_dir).resolve()) or meta['metadata'] != str(Path(metadata).resolve())
                or meta['settings'] != settings):
            raise ValueError('Different extraction inputs; use a new --output-root')
        print(f'D Stage 2: verified {meta["n_patches"]} cached patches')
        return meta
    info = inventory(pool_dir, metadata)
    sources = source_snapshot(pool_dir, metadata)
    split = common.spatial_split(sources, settings['cell_m'], settings['buffer_m'],
                               settings['val_fraction'], settings['seed'])
    plan = dict(stage=2, adapter='portable_pool', pool_dir=info['pool_dir'], metadata=info['metadata'],
                metadata_sha256=sha(metadata), sources=sources, settings=settings, split=split,
                inventory=info, code_sha256=code_hashes())
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
