"""Before/after regression checks. No downloads, SAM weights or AE training.

The .txt fixtures freeze the actual pre-refactor source, not a rewritten copy
of the new functions. Execute these tests before reusing existing artifacts.
"""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

from aibip import config, part_b, c_transfer, d_data, d_pool, pool_data
from aibip import b_refactor_compat as compat

FIXTURES = Path(__file__).parent/'fixtures/before_b_refactor'


def before(name):
    module = ModuleType('before_'+name.removesuffix('.py'))
    module.__file__ = str(FIXTURES/(name+'.txt'))
    exec(compile((FIXTURES/(name+'.txt')).read_bytes(), module.__file__, 'exec'), module.__dict__)
    return module


def test_reference_sources_match_recorded_pre_refactor_hashes():
    registry = compat._approved_registry()
    assert registry is not None
    for name in ['part_b.py','c_transfer.py','c_test_core.py']:
        source = (FIXTURES/(name+'.txt')).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(source).hexdigest() in registry['legacy_sha256'][name]


@pytest.mark.parametrize('shape', [(83,129,3), (721,1003,3), (1000,1200,3)])
def test_b_rgb_work_pixels_and_projection_match_frozen_sources(shape, monkeypatch):
    old_b, old_c = before('part_b.py'), before('c_transfer.py')
    monkeypatch.setattr(config, 'WORK_WIDTH', 320)
    raw = np.random.default_rng(4).integers(0,256,shape,dtype=np.uint8)
    old_crop = old_c.crop_rgb(raw)
    h, w = old_crop.shape[:2]
    small_w = min(w,1024)
    expected = cv2.resize(old_crop,(small_w,max(1,round(h*small_w/w))),interpolation=cv2.INTER_AREA)
    actual, geo = part_b.preprocess_for_segmentation(raw)
    assert np.array_equal(actual, expected)
    assert np.array_equal(c_transfer.crop_rgb(raw), old_crop)
    assert geo == old_c.geometry(shape)
    for clahe in [False, True]:
        monkeypatch.setattr(config, 'USE_CLAHE', clahe)
        assert np.array_equal(part_b.preprocess(raw), old_b.preprocess(raw))
    mask = np.random.default_rng(5).integers(0,2,actual.shape[:2]).astype(bool)
    assert np.array_equal(part_b.mask_to_work_view(mask,shape),old_c.project_mask(mask,shape))


class Predictor:
    def __init__(self):
        self.calls = []
    def set_image(self, image):
        self.image = image.copy()
    def predict(self, **kwargs):
        self.calls.append(kwargs)
        h,w = self.image.shape[:2]
        road = np.zeros((h,w),bool)
        road[h//4:, w//6:5*w//6] = True
        return np.stack([road, road, np.zeros_like(road)]),np.array([.95,.8,.6]),None


def test_full_c_path_pixels_prompts_candidates_and_masks_match(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(inference_mode=nullcontext,set_num_threads=lambda n:None))
    raw = np.full((1000,1200,3),110,np.uint8)
    raw[500:600,200:350] = [30,140,30]
    old_c, old_core = before('c_transfer.py'), before('c_test_core.py')
    old_predictor, new_predictor = Predictor(), Predictor()
    old_c.TestRoadSegmenter = old_core.TestRoadSegmenter
    old_c._api = lambda:SimpleNamespace(_get_predictor=lambda backend:old_predictor)
    monkeypatch.setattr(c_transfer,'_api',lambda:SimpleNamespace(_get_predictor=lambda backend:new_predictor))
    expected = old_c.segment_details(raw)
    actual = c_transfer.segment_details(raw)
    assert np.array_equal(old_predictor.image,new_predictor.image)
    for old, new in zip(old_predictor.calls,new_predictor.calls):
        assert old['multimask_output'] == new['multimask_output']
        assert np.array_equal(old['point_coords'],new['point_coords'])
        assert np.array_equal(old['point_labels'],new['point_labels'])
    assert len(new_predictor.calls) == 5
    for i in range(3):
        assert np.array_equal(expected[i],actual[i])
    assert expected[3] == actual[3]


def test_d_patch_pixels_positions_and_work_mask_are_identical(tmp_path,monkeypatch):
    old_b, old_c = before('part_b.py'), before('c_transfer.py')
    monkeypatch.setattr(config,'WORK_WIDTH',800)
    raw = np.random.default_rng(6).integers(20,200,(1000,800,3),dtype=np.uint8)
    mask = old_c.project_mask(np.ones((280,800),bool),raw.shape)
    source, mask_path = tmp_path/'road.png',tmp_path/'mask.png'
    cv2.imwrite(str(source),raw)
    cv2.imwrite(str(mask_path),mask.astype(np.uint8)*255)
    src = dict(source=str(source),mask=str(mask_path),work_shape=list(mask.shape),
               original_shape=list(raw.shape),image_id='road',c_status='ok')
    settings = dict(cap=24,spacing=64,seed=42,filter_mode='none',appearance='clahe')
    array,centers,_ = d_pool.extract_image(src,settings)
    expected_centers = d_data.spaced_centers(mask,'road',24,64,42)
    old_work = old_b.apply_clahe(old_b.resize_band(old_b.crop_roi(raw)))
    expected = np.stack([old_work[c['row']-64:c['row']+64,c['col']-64:c['col']+64] for c in expected_centers])
    assert len(array) > 0 and centers == expected_centers
    assert np.array_equal(array,expected)


def old_codes(current):
    registry = compat._approved_registry()
    assert registry is not None, 'Source bundle changed: do not silently refresh the registry'
    return {k:registry['legacy_sha256'].get(k,[v])[0] for k,v in current.items() if k!='b_refactor_compat'}


@pytest.mark.parametrize('get_codes',[d_data.data_code_hashes,d_pool.code_hashes,pool_data.code_hashes])
def test_existing_d_manifest_code_identity_is_accepted_without_rewriting(get_codes):
    current = get_codes()
    old = old_codes(current)
    assert compat.compatible_code_hashes(old,current)
    assert compat.compatible_code_hashes(current,current)
    bad = dict(old,part_b='unrecognized')
    assert not compat.compatible_code_hashes(bad,current)
    for name in old:
        assert not compat.compatible_code_hashes(dict(old,**{name:'0'*64}),current)


def c_keys(tmp_path,monkeypatch):
    weights = tmp_path/'mobile_sam_vit_t.pt'
    weights.write_bytes(b'fake weights; no model is loaded')
    monkeypatch.setattr(config,'MODELS_DIR',tmp_path)
    key,settings = c_transfer._run_key('mobile_sam')
    old_settings = dict(settings,code_sha256=old_codes(settings['code_sha256']))
    old_key = hashlib.sha256(json.dumps(old_settings,sort_keys=True).encode()).hexdigest()
    return key,settings,old_key


def test_old_mask_sidecar_is_read_only_and_changed_contents_still_fail(tmp_path,monkeypatch):
    key,settings,old_key = c_keys(tmp_path,monkeypatch)
    allowed = compat.compatible_run_keys(key,settings)
    assert old_key in allowed
    mask,sidecar,preview = [tmp_path/name for name in ['road.png','road.test_c.json','road.jpg']]
    cv2.imwrite(str(mask),np.full((12,24),255,np.uint8))
    preview.write_bytes(b'preview fixture')
    record = dict(cache_key='source'+old_key,mask_sha256=c_transfer._hash(mask),target_shape=[12,24])
    sidecar.write_text(json.dumps(record))
    saved = sidecar.read_bytes()
    assert c_transfer._cached(mask,sidecar,preview,'source'+key,{'source'+k for k in allowed}) == record
    assert sidecar.read_bytes() == saved
    assert c_transfer._cached(mask,sidecar,preview,'changed'+key,{'changed'+k for k in allowed}) is None
    cv2.imwrite(str(mask),np.zeros((12,24),np.uint8))
    assert c_transfer._cached(mask,sidecar,preview,'source'+key,{'source'+k for k in allowed}) is None
    changed = dict(settings,c_crop=[.45,.72])
    assert old_key not in compat.compatible_run_keys('new-run',changed)
    changed = dict(settings,weights_sha256='other weights')
    assert old_key not in compat.compatible_run_keys('new-run',changed)


def test_source_edit_disables_legacy_acceptance(tmp_path,monkeypatch):
    current = d_data.data_code_hashes()
    old = old_codes(current)
    registry = compat._approved_registry()
    for name in [*registry['approved_lf_sha256'],compat.REGISTRY_NAME]:
        shutil.copy2(compat.PACKAGE/name,tmp_path/name)
    # Even an edit to a moved helper must invalidate historical acceptance.
    with (tmp_path/'part_b.py').open('a') as file:
        file.write('\n# simulated subsequent edit\n')
    monkeypatch.setattr(compat,'PACKAGE',tmp_path)
    assert compat._approved_registry() is None
    assert not compat.compatible_code_hashes(old,current)


def test_completed_legacy_patch_manifest_is_not_modified(tmp_path,monkeypatch):
    shard,index,manifest = pool_data.paths(tmp_path)
    shard.parent.mkdir()
    shard.write_bytes(b'shard bytes')
    index.write_bytes(b'index bytes')
    metadata = tmp_path/'metadata.json'
    metadata.write_text('[]')
    meta = dict(stage=2,adapter='portable_pool',code_sha256=old_codes(pool_data.code_hashes()),
                metadata=str(metadata),metadata_sha256=d_data.sha(metadata),pool_dir=str(tmp_path),sources=[],
                shard_sha256=d_data.sha(shard),index_sha256=d_data.sha(index))
    d_data.save_json(manifest,meta)
    saved = manifest.read_bytes()
    monkeypatch.setattr(pool_data,'source_snapshot',lambda *a:[])
    assert pool_data.validate(tmp_path)==meta
    assert manifest.read_bytes()==saved  # checkpoint's manifest SHA remains valid
    shard.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='incomplete'):
        pool_data.validate(tmp_path)


def test_partial_plan_requires_same_settings_and_sources():
    current = dict(code_sha256=pool_data.code_hashes(),sources=['same'],settings=dict(cap=24))
    old = dict(current,code_sha256=old_codes(current['code_sha256']))
    assert compat.compatible_plan(old,current)
    assert not compat.compatible_plan(old,dict(current,settings=dict(cap=12)))
    assert not compat.compatible_plan(old,dict(current,sources=['changed']))


def test_three_filter_rows_and_standalone_command(tmp_path,monkeypatch):
    images = [tmp_path/f'i{i}.jpg' for i in range(3)]
    for p in images:
        p.write_bytes(b'not decoded in this test')
    monkeypatch.setattr(part_b.imgio,'load_image',lambda path:np.full((100,200,3),100,np.uint8))
    monkeypatch.setattr(part_b,'preprocess',lambda image:image)
    observed=[]
    from matplotlib.figure import Figure
    def save(fig,*a,**kw):
        observed.append(fig)
        assert len(fig.axes)==12
        assert all(p.stem in fig.axes[i*4].get_title() for i,p in enumerate(images))
    monkeypatch.setattr(Figure,'savefig',save)
    part_b.figure_filters(images,tmp_path/'out.png')
    assert len(observed)==1
    calls=[]
    monkeypatch.setattr(part_b,'figure_filters',lambda paths,out:calls.append((paths,out)))
    part_b.main(['filters',str(tmp_path),'--output',str(tmp_path/'filters.png')])
    assert len(calls)==1 and len(calls[0][0])==3
