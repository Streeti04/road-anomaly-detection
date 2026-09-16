"""Portable pool contract: no Mini assets, recursive files, resume and dispatch."""
import cv2
import numpy as np
import pytest

from aibip import config, d_data, pool_data, pool, part_d, d_training


def test_inventory_works_without_any_mini_files(tmp_path, monkeypatch):
    images = tmp_path/'images'
    (images/'nested').mkdir(parents=True)
    (images/'a.jpg').write_bytes(b'a')
    (images/'nested/b.PNG').write_bytes(b'b')
    metadata = tmp_path/'recordings.json'
    d_data.save_json(metadata, [dict(imageId=k,latitude=50,longitude=8) for k in ['a','b','missing']])
    monkeypatch.setattr(config,'MINI_DIR',tmp_path/'does_not_exist')
    monkeypatch.setattr(d_data,'validate',lambda *a: pytest.fail('Must not access the Mini patch manifest'))
    info = pool_data.inventory(images,metadata)
    assert info['n_images'] == 2 and info['image_ids'] == ['a','b']
    assert info['missing_image_ids'] == ['missing']
    assert 'mini_dir' not in info and 'mini_ids' not in info
    (images/'nested/a.png').write_bytes(b'another a')
    with pytest.raises(ValueError,match='Duplicate'):
        pool_data.inventory(images,metadata)


def test_coordinates_are_required_instead_of_patch_random_split(tmp_path):
    (tmp_path/'a.jpg').write_bytes(b'a')
    (tmp_path/'b.jpg').write_bytes(b'b')
    metadata=tmp_path/'meta.json'
    d_data.save_json(metadata,[dict(image_id='a',latitude=50,longitude=8)])
    with pytest.raises(ValueError,match='Missing coordinates'):
        pool_data.inventory(tmp_path,metadata)


def test_optional_evaluation_subset_checks_identity(tmp_path):
    (tmp_path/'a.jpg').write_bytes(b'a')
    meta=dict(sources=[dict(image_id='a',source_sha256=d_data.sha(tmp_path/'a.jpg')),
                       dict(image_id='b',source_sha256='b')])
    assert pool.evaluation_ids(meta)=={'a','b'}
    assert pool.evaluation_ids(meta,tmp_path)=={'a'}
    (tmp_path/'a.jpg').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='unchanged'):
        pool.evaluation_ids(meta,tmp_path)


def test_cli_routes_pool_without_mini_reference(monkeypatch,tmp_path):
    monkeypatch.chdir(pool_data.ROOT)
    calls=[]
    monkeypatch.setattr(pool_data,'extract',lambda **kw:calls.append(kw))
    pool.main(['extract','--images',str(tmp_path/'images'),'--metadata',str(tmp_path/'meta.json')])
    assert calls[0]['pool_dir']==tmp_path/'images' and 'mini_reference' not in calls[0]
    assert calls[0]['cap']==24 and calls[0]['cell_m']==300.
    monkeypatch.setattr(d_training,'train',lambda **kw:calls.append(kw))
    pool.main(['train','--device','cpu'])
    assert calls[1]['stage']==2 and calls[1]['data_module'] is pool_data
    assert calls[1]['output_root']==pool_data.DEFAULT_OUTPUT


def test_portable_extraction_resumes_and_uses_identical_sampler(tmp_path,monkeypatch):
    images=tmp_path/'images'
    images.mkdir()
    monkeypatch.setattr(config,'WORK_WIDTH',256)
    monkeypatch.setattr(config,'MINI_DIR',tmp_path/'no_mini')
    monkeypatch.setattr(d_data,'validate',lambda *a:pytest.fail('Mini must not be validated'))
    sources=[]
    for i in range(3):
        source,mask=images/f'i{i}.jpg',tmp_path/f'mask{i}.png'
        cv2.imwrite(str(source),np.full((1000,256,3),100,np.uint8))
        cv2.imwrite(str(mask),np.full((400,256),255,np.uint8))
        sources.append(dict(image_id=f'i{i}',source=str(source),mask=str(mask),c_status='ok',
                            work_shape=[400,256],original_shape=[1000,256,3]))
    metadata=tmp_path/'metadata.json'
    d_data.save_json(metadata,[dict(image_id=f'i{i}',latitude=50+i*.01,longitude=8) for i in range(3)])
    monkeypatch.setattr(pool_data,'source_snapshot',lambda *a:sources)
    split=dict(train_ids=['i0'],val_ids=['i1'],buffer_ids=['i2'])
    monkeypatch.setattr(d_data,'spatial_split',lambda *a:split)
    actual=pool_data.extract_image
    def interrupted(src,settings):
        if src['image_id']=='i1':
            raise RuntimeError('interruption')
        return actual(src,settings)
    monkeypatch.setattr(pool_data,'extract_image',interrupted)
    root=tmp_path/'output'
    with pytest.raises(RuntimeError,match='interruption'):
        pool_data.extract(images,metadata,root)
    assert (root/'chunks/i0.json').exists()
    calls=[]
    def resumed(src,settings):
        calls.append(src['image_id'])
        return actual(src,settings)
    monkeypatch.setattr(pool_data,'extract_image',resumed)
    meta=pool_data.extract(images,metadata,root)
    assert calls==['i1','i2'] and meta['adapter']=='portable_pool'
    assert 'mini_manifest_sha256' not in meta
    assert pool_data.validate(root)==meta
    array,centers,_=actual(sources[0],pool_data.extraction_settings())
    shard,index,_=pool_data.paths(root)
    assert np.array_equal(np.load(shard)[:len(array)],array)
    assert [dict(band=r['band'],row=r['row'],col=r['col']) for r in part_d.read_index(index)[:len(array)]]==centers
    with pytest.raises(ValueError,match='Different extraction'):
        pool_data.extract(images,metadata,root,cap=12)


def test_output_cannot_pollute_recursive_source_directory(tmp_path):
    with pytest.raises(ValueError,match='outside'):
        pool_data.extract(tmp_path,tmp_path/'metadata.json',tmp_path/'results')


def test_historical_results_are_protected():
    for root in [d_data.DEFAULT_OUTPUT,pool_data.ROOT/'outputs/d_stage2']:
        with pytest.raises(ValueError,match='historical'):
            pool_data.paths(root)


def test_ranking_uses_median_and_flags_low_support():
    rows=[dict(image_id='many') for _ in range(5)]+[dict(image_id='two') for _ in range(2)]+[dict(image_id='one')]
    errors=[.20,.02,.02,.02,.02, .09,.09, .05]
    split=dict(train_ids=['many','two','one'],val_ids=[],buffer_ids=[])
    ranking,_=pool.ranked_images(['many','two','one','ghost'],rows,errors,split)
    assert [r['image_id'] for r in ranking]==['many','one','two','ghost']
    many,one,two,ghost=ranking
    assert many['median_error']==pytest.approx(.02) and many['mean_error']==pytest.approx(.056)
    assert many['n_patches']==5 and many['status']=='scored' and many['rank_typical_to_anomalous']==1
    # Single-patch image flips from less to more anomalous vs the old mean signal.
    assert one['median_error']==pytest.approx(.05) and one['n_patches']==1
    assert one['status']=='low_support' and one['rank_typical_to_anomalous']==2
    assert two['status']=='low_support' and two['median_error']==pytest.approx(.09)
    assert ghost['status']=='no_patches' and ghost['median_error'] is None
    assert ghost['rank_typical_to_anomalous'] is None
