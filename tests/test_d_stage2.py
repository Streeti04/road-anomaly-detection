"""Pool adapter tests; synthetic images only, no SAM or real dataset training."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aibip import d_data, d_pool, d_stage2, d_training, config, part_d


def test_stage2_paths_cannot_overwrite_mini():
    for root in [d_data.DEFAULT_OUTPUT, d_data.DEFAULT_OUTPUT/'child', d_data.DEFAULT_OUTPUT.parent]:
        with pytest.raises(ValueError, match='separate'):
            d_pool.paths(root)
    assert d_pool.paths(d_pool.DEFAULT_OUTPUT)[0].name == 'patches_stage2.npy'
    assert d_training.run_path('outputs/d_stage2', 'mse', 2).name == 'stage2'
    assert d_training.run_path('outputs/d_v2', 'mse').name == 'stage1'


def test_inventory_counts_files_and_checks_mini_bytes(tmp_path):
    pool, mini = tmp_path/'pool', tmp_path/'mini'
    pool.mkdir()
    mini.mkdir()
    for key in ['a','b','c']:
        (pool/f'{key}.jpg').write_bytes(key.encode())
    (mini/'a.jpg').write_bytes(b'a')
    metadata = tmp_path/'metadata.json'
    d_data.save_json(metadata, [dict(image_id=key, latitude=50, longitude=8) for key in ['a','b','c','missing']])
    result = d_pool.inventory(pool, metadata, mini)
    assert result['n_images'] == 3 and result['n_mini'] == 1
    assert result['missing_image_ids'] == ['missing']
    (mini/'a.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError, match='different bytes'):
        d_pool.inventory(pool, metadata, mini)
    with pytest.raises(ValueError, match='larger'):
        d_pool.inventory(mini, metadata, mini)


def test_rank_means_and_keep_empty_images_without_numbers():
    rows = [dict(image_id='a'),dict(image_id='a'),dict(image_id='b')]
    split = dict(train_ids=['a'], val_ids=['b'], buffer_ids=['c'])
    result, _ = d_stage2.ranked_images(['a','b','c'], rows, [.01,.03,.01], split)
    assert [r['image_id'] for r in result] == ['b','a','c']
    assert result[1]['mean_error'] == .02 and result[1]['max_error'] == .03
    assert result[2]['mean_error'] is None and result[2]['rank_typical_to_anomalous'] is None
    assert result[2]['split'] == 'buffer' and result[2]['status'] == 'no_patches'
    with pytest.raises(ValueError, match='finite'):
        d_stage2.ranked_images(['a'], [dict(image_id='a')], [np.nan], split)


def test_pool_extraction_resumes_completed_images_and_validates_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'WORK_WIDTH', 256)
    settings = dict(cap=24, spacing=64, filter_mode='green', appearance='original',
                    cell_m=300., buffer_m=30., val_fraction=.2, seed=42,
                    patch=128, bands=4, width=256, roi=[.4,.8])
    sources = []
    for i in range(3):
        image, mask = tmp_path/f'i{i}.png', tmp_path/f'm{i}.png'
        raw = np.full((1000,256,3), 100, np.uint8)
        if i == 2:
            raw[:] = [30,140,30]
        cv2.imwrite(str(image), raw)
        cv2.imwrite(str(mask), np.full((400,256),255,np.uint8))
        sources.append(dict(image_id=f'i{i}', source=str(image), mask=str(mask),
                            c_status='ok', work_shape=[400,256], original_shape=[1000,256,3]))
    metadata = tmp_path/'metadata.json'
    metadata.write_text('[]')
    reference = tmp_path/'reference'
    reference_manifest = d_data.paths(reference)[2]
    d_data.save_json(reference_manifest, dict(settings=settings))
    monkeypatch.setattr(d_data, 'validate', lambda path: dict(settings=settings))
    split = dict(train_ids=['i0'],val_ids=['i1'],buffer_ids=['i2'])
    monkeypatch.setattr(d_data, 'spatial_split', lambda *a: split)
    monkeypatch.setattr(d_pool, 'inventory', lambda *a: dict(pool_dir=str(tmp_path.resolve()),
        metadata=str(metadata.resolve()), mini_ids=['i0'], n_images=3, n_mini=1))
    monkeypatch.setattr(d_pool, 'source_snapshot', lambda *a: sources)
    actual = d_pool.extract_image
    calls = []
    def interrupted(src, settings):
        calls.append(src['image_id'])
        if src['image_id'] == 'i1':
            raise RuntimeError('simulated interruption')
        return actual(src, settings)
    monkeypatch.setattr(d_pool, 'extract_image', interrupted)
    root = tmp_path/'stage2'
    kwargs = dict(pool_dir=tmp_path, metadata=metadata, output_root=root, mini_reference=reference)
    with pytest.raises(RuntimeError, match='interruption'):
        d_pool.extract(**kwargs)
    assert (root/'chunks/i0.json').exists() and not d_pool.paths(root)[2].exists()
    calls.clear()
    def resumed(src, settings):
        calls.append(src['image_id'])
        return actual(src, settings)
    monkeypatch.setattr(d_pool, 'extract_image', resumed)
    meta = d_pool.extract(**kwargs)
    assert calls == ['i1','i2']
    assert meta['stage'] == 2 and meta['audit'][2]['status'] == 'all_filtered'
    assert d_pool.validate(root) == meta
    shard, index, manifest = d_pool.paths(root)
    patches = np.load(shard)
    rows = part_d.read_index(index)
    assert patches.shape == (len(rows),128,128,3)
    # Adapter must use exactly the same Mini centers and pixel preprocessing.
    expected, centers, _ = actual(sources[0], settings)
    assert np.array_equal(patches[:len(expected)], expected)
    assert [{k:r[k] for k in ['row','col','band']} for r in rows[:len(expected)]] == centers
    settings['spacing'] = 80
    d_data.save_json(reference_manifest, dict(settings=settings))
    with pytest.raises(ValueError, match='Different extraction'):
        d_pool.extract(**kwargs)
    shard.write_bytes(b'changed')
    with pytest.raises(ValueError, match='incomplete'):
        d_pool.validate(root)


def test_stage2_model_checkpoint_and_preview(tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(tmp_path)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr


def worker(root):
    import torch
    from types import SimpleNamespace
    torch.set_num_threads(1)
    part_d._build_model = lambda: torch.nn.Sequential(torch.nn.Conv2d(3,3,1), torch.nn.Sigmoid())
    shard, index, manifest = d_pool.paths(root)
    shard.parent.mkdir(parents=True)
    np.save(shard, np.random.default_rng(1).integers(20,200,(4,128,128,3),dtype=np.uint8))
    rows = [dict(patch_idx=i,image_id='a' if i<2 else 'b',band=1,row=100,col=100) for i in range(4)]
    part_d.write_index(index, rows)
    meta = dict(stage=2, split=dict(train_ids=['a'],val_ids=['b'],buffer_ids=[]),
                sources=[dict(image_id='a'),dict(image_id='b')])
    d_data.save_json(manifest, meta)
    adapter = SimpleNamespace(STAGE=2, validate=lambda p: meta, paths=d_pool.paths,
                              sha=d_data.sha, read_json=d_data.read_json, save_json=d_data.save_json)
    # Production loop and preview, tiny model only. Fresh initialization is seeded.
    kwargs = dict(stage=2, output_root=root, data_module=adapter, batch_size=2, threads=1, patience=5)
    d_training.train(epochs=1, run_name='first', **kwargs)
    d_training.train(epochs=1, run_name='fresh', **kwargs)
    load = lambda name: torch.load(d_training.run_path(root,name,2)/'best.pt', weights_only=True)
    first, fresh = load('first'), load('fresh')
    assert first['stage'] == 2
    assert all(torch.equal(v, fresh['state_dict'][k]) for k,v in first['state_dict'].items())
    assert not d_training.run_path(root,'first',1).exists()
    assert (d_training.run_path(root,'first',2)/'validation_preview.jpg').exists()
    d_training.train(epochs=2, run_name='first', resume=True, **kwargs)
    last = torch.load(d_training.run_path(root,'first',2)/'last.pt', weights_only=True)
    assert last['epoch'] == 2 and len(last['history']) == 2
    try:
        d_training.train(epochs=31, run_name='too_long', **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError('Stage 2 must keep the 30-epoch limit')


if __name__ == '__main__':
    # Make project imports available when invoked as a subprocess script.
    worker(Path(sys.argv[2]))
