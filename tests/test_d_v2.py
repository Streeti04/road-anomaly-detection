"""D data regression tests; no training or model downloads."""
import numpy as np
import pytest

from aibip import part_d
from aibip import d_data as data


def test_square_patch_corners_and_holes_are_inside_road():
    yy, xx = np.mgrid[:600, :900]
    mask = (xx > yy) & (xx < 860)
    mask[240:265, 610:630] = False
    centers = part_d.plan_centers(mask, 'diagonal')
    assert centers
    for c in centers:
        y, x = c['row'], c['col']
        assert mask[y-64:y+64, x-64:x+64].shape == (128, 128)
        assert mask[y-64:y+64, x-64:x+64].all()


def test_empty_and_small_mask_return_no_centers():
    assert part_d.plan_centers(np.zeros((614, 2048), bool), 'empty') == []
    assert part_d.plan_centers(np.ones((90, 90), bool), 'tiny') == []


@pytest.mark.parametrize('weights,total', [([600,300,90,10],24),([3,1000,0,900],24),
                                           ([1,1,1,1],3),([10,1000,1000],10000),([0,0],24)])
def test_budget_counts_each_remainder_once(weights, total):
    allocation = part_d.alloc_budget(np.array(weights), total)
    assert allocation.sum() == min(total, sum(weights))
    assert (allocation >= 0).all() and (allocation <= weights).all()
    with pytest.raises(ValueError):
        part_d.alloc_budget(np.array([-1,2]), 5)


def test_sampling_has_spacing_and_is_repeatable():
    mask = np.zeros((614, 2048), bool)
    mask[80:530, 100:1900] = True
    centers = data.spaced_centers(mask, 'sample', cap=24, spacing=64)
    assert 1 < len(centers) <= 24
    assert centers == data.spaced_centers(mask, 'sample', cap=24, spacing=64)
    for i, a in enumerate(centers):
        for b in centers[i+1:]:
            assert (a['row']-b['row'])**2+(a['col']-b['col'])**2 >= 64**2


def test_spatial_groups_and_exact_duplicates_never_cross_split():
    records = [dict(image_id=f'i{i}', source_sha256=f'h{i}',
                    coordinates=[50.+i*.01, 8.+i*.01]) for i in range(12)]
    records[1]['source_sha256'] = records[0]['source_sha256']
    split = data.spatial_split(records, cell_m=100, buffer_m=30, seed=42)
    assert split == data.spatial_split(list(reversed(records)), cell_m=100, buffer_m=30, seed=42)
    train, val, buffer = [set(split[k]) for k in ['train_ids','val_ids','buffer_ids']]
    assert train and val and not train.intersection(val)
    assert not buffer.intersection(train | val)
    assert train | val | buffer == {r['image_id'] for r in records}
    assert ('i0' in val) == ('i1' in val)
    assert any({'i0','i1'} <= set(group) for group in split['groups'])


def test_spatial_buffer_removes_close_neighbors_across_cell_boundary():
    # Deliberately tiny cells; the 10 m pair lies in different cells but must
    # never appear on opposite sides of a split with a 30 m buffer.
    records = [dict(image_id=f'i{i}', source_sha256=f'h{i}',
                    coordinates=[50.+offset/111320, 8.])
               for i, offset in enumerate([0, 10, 1000, 2000, 3000, 4000])]
    for seed in range(10):
        split = data.spatial_split(records, cell_m=2, buffer_m=30, seed=seed)
        assert not ({'i0','i1'} & set(split['val_ids']) and {'i0','i1'} & set(split['train_ids']))


def test_single_group_fails_instead_of_random_patch_split():
    records = [dict(image_id=str(i), source_sha256=str(i), coordinates=[50,8]) for i in range(3)]
    with pytest.raises(ValueError, match='one spatial'):
        data.spatial_split(records)


def test_green_filter_does_not_remove_plain_brown_or_gray_road():
    gray = np.full((128,128,3), 190, np.uint8)
    brown = np.full((128,128,3), [60,115,145], np.uint8)
    grass = np.full((128,128,3), [30,140,30], np.uint8)
    assert data.filter_reason(gray, 'green') == ''
    assert data.filter_reason(brown, 'green') == ''
    assert data.filter_reason(grass, 'green')
    assert data.filter_reason(grass, 'none') == ''
    assert data.filter_reason(brown, 'legacy')


def test_legacy_random_split_helper_keeps_training_nonempty():
    train, val = part_d.split_ids(['a','b'], val_frac=.99)
    assert len(train) == len(val) == 1


def test_changed_shard_and_changed_sources_are_rejected(tmp_path, monkeypatch):
    shard, index, manifest = data.paths(tmp_path)
    shard.parent.mkdir()
    np.save(shard, np.zeros((1,128,128,3), np.uint8))
    index.write_text('index fixture')
    metadata = tmp_path/'metadata.json'
    metadata.write_text('[]')
    monkeypatch.setattr(data, 'data_code_hashes', lambda: {})
    monkeypatch.setattr(data, 'source_snapshot', lambda path: [])
    meta = dict(code_sha256={}, sources=[], metadata=str(metadata), metadata_sha256=data.sha(metadata),
                shard_sha256=data.sha(shard), index_sha256=data.sha(index))
    data.save_json(manifest, meta)
    assert data.validate(tmp_path) == meta
    monkeypatch.setattr(data, 'source_snapshot', lambda path: ['changed'])
    with pytest.raises(ValueError, match='changed'):
        data.validate(tmp_path)
    monkeypatch.setattr(data, 'source_snapshot', lambda path: [])
    shard.write_bytes(b'changed')
    with pytest.raises(ValueError, match='incomplete'):
        data.validate(tmp_path)


def test_stage2_disabled_and_cli_controls_forwarded(monkeypatch):
    from aibip import d_training
    calls = []
    monkeypatch.setattr(d_training, 'train', lambda **kw: calls.append(kw))
    d_training.main(['train','--run-name','edge','--edge-weight','.1','--epochs','2'])
    assert calls[0]['edge_weight'] == .1 and calls[0]['epochs'] == 2
    with pytest.raises(ValueError):
        data.require_mini(2)


def test_extract_writes_current_mask_cache_and_separates_all_filtered(tmp_path, monkeypatch):
    import cv2
    from aibip import config
    monkeypatch.setattr(config, 'WORK_WIDTH', 256)
    sources = []
    for i in range(3):
        image_path, mask_path = tmp_path/f'image{i}.png', tmp_path/f'mask{i}.png'
        image = np.full((1000,256,3), 100, np.uint8)
        if i == 2:
            image[:] = [30,140,30]
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(mask_path), np.full((400,256), 255, np.uint8))
        sources.append(dict(image_id=f'i{i}', source=str(image_path), mask=str(mask_path),
                            c_status='ok', work_shape=[400,256], original_shape=[1000,256,3]))
    metadata = tmp_path/'metadata.json'
    metadata.write_text('[]')
    monkeypatch.setattr(data, 'source_snapshot', lambda path: sources)
    monkeypatch.setattr(data, 'spatial_split', lambda *a: dict(train_ids=['i0'],val_ids=['i1'],buffer_ids=['i2']))
    root = tmp_path/'out'
    meta = data.extract(output_root=root, metadata=metadata, appearance='original')
    assert meta['n_patches'] > 1
    assert meta['audit'][2]['status'] == 'all_filtered'
    assert meta['audit'][2]['rejected']['green_vegetation_hint'] > 0
    assert all(r['image_id'] != 'i2' for r in part_d.read_index(data.paths(root)[1]))
    assert (root/'patches_rejected.jpg').exists()
