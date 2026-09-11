"""Transfer regression tests; no model downloads, training or real SAM inference."""
from contextlib import nullcontext
from types import SimpleNamespace
import json
import sys

import cv2
import numpy as np
import pytest

from aibip import config, part_c, part_b
from aibip import c_transfer as transfer
from aibip.c_test_core import TestRoadSegmenter, vegetation, seed_component, candidate_score


def test_entry_points_use_transferred_implementation():
    assert part_c.segment_road is transfer.segment_road
    assert part_c.run is transfer.run
    assert part_c.single is transfer.single
    assert part_c.qc is transfer.qc
    assert part_c.benchmark is transfer.benchmark


def test_crop_is_unenhanced_rgb_and_independent_of_B(monkeypatch):
    image = np.zeros((1000, 800, 3), np.uint8)
    image[450:730] = [20, 70, 180]
    monkeypatch.setattr(config, 'USE_CLAHE', True)
    crop = transfer.crop_rgb(image)
    assert crop.shape == (280, 800, 3)
    assert (crop == [180, 70, 20]).all()
    with pytest.raises(ValueError):
        transfer.crop_rgb(image.astype(float))


def test_projection_matches_explicit_native_mapping(monkeypatch):
    monkeypatch.setattr(config, 'WORK_WIDTH', 320)
    monkeypatch.setattr(config, 'ROI_TOP_FRAC', .40)
    monkeypatch.setattr(config, 'ROI_BOTTOM_FRAC', .80)
    shape = (800, 640, 3)
    small = np.zeros((14, 40), bool)
    small[3:, 8:32] = True
    native = np.zeros(shape[:2], np.uint8)
    native[360:584] = cv2.resize(small.astype(np.uint8), (640, 224), interpolation=cv2.INTER_NEAREST)
    expected = cv2.resize(native[320:640], (320, 160), interpolation=cv2.INTER_NEAREST).astype(bool)
    actual = transfer.project_mask(small, shape)
    assert actual.dtype == bool and actual.shape == (160, 320)
    assert np.array_equal(actual, expected)
    assert not actual[:20].any() and not actual[132:].any()


def test_original_resolution_geometry():
    geo = transfer.geometry((6144, 8192, 3))
    assert geo['crop_bounds'] == [0, 2764, 8192, 4485]
    assert geo['target_shape'] == [614, 2048]
    mask = transfer.project_mask(np.ones((215, 1024), bool), (6144, 8192, 3))
    assert mask[77:507].all()
    assert not mask[:77].any() and not mask[507:].any()


def test_connected_seed_and_vegetation_penalty():
    mask = np.zeros((50, 100), bool)
    mask[10:, 10:30] = True
    mask[20:, 60:90] = True
    component = seed_component(mask, [(20, 40)])
    assert component[40, 20] and not component[40, 70]
    green = vegetation(np.full((50, 100, 3), [40, 140, 40], np.uint8))
    plain = np.zeros((50, 100), bool)
    road_score, _ = candidate_score(component, .9, plain, [(20, 40)])
    grass_score, _ = candidate_score(component, .9, green, [(20, 40)])
    assert road_score > grass_score
    assert not seed_component(mask, [(0, 0)]).any()


def test_five_prompts_and_bool_mask_without_loading_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext))
    class Predictor:
        def __init__(self):
            self.calls = []
        def set_image(self, image):
            self.shape = image.shape[:2]
        def predict(self, **kwargs):
            self.calls.append(kwargs)
            h, w = self.shape
            mask = np.zeros((h, w), bool)
            mask[h//4:, w//5:4*w//5] = True
            return np.stack([mask]*3), np.array([.9, .8, .7]), None
    predictor = Predictor()
    image = np.full((280, 1000, 3), 100, np.uint8)
    full, small, _, info = TestRoadSegmenter(predictor).predict(image)
    assert full.shape == image.shape[:2] and full.dtype == bool
    assert info['status'] == 'ok' and len(info['candidates']) == 15
    assert len(predictor.calls) == 5
    assert [c['point_coords'][0, 0] for c in predictor.calls] == [200, 350, 500, 650, 800]
    assert all(c['multimask_output'] for c in predictor.calls)


def test_empty_candidates_return_no_road(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext))
    predictor = SimpleNamespace(set_image=lambda image: None,
        predict=lambda **kwargs: (np.zeros((3, 28, 100), bool), np.ones(3), None))
    full, _, _, info = TestRoadSegmenter(predictor).predict(np.full((28, 100, 3), 100, np.uint8))
    assert info['status'] == 'no_road' and not full.any()


def configure_outputs(tmp_path, monkeypatch):
    output = tmp_path/'outputs'
    monkeypatch.setattr(config, 'OUTPUTS_DIR', output)
    monkeypatch.setattr(config, 'MASKS_DIR', output/'masks')
    monkeypatch.setattr(config, 'PATCHES_DIR', output/'patches')
    monkeypatch.setattr(config, 'FLAGS_CSV', output/'flags.csv')
    monkeypatch.setattr(transfer, '_run_key', lambda backend: ('code-key', {}))
    source = tmp_path/'test.png'
    cv2.imwrite(str(source), np.full((1000, 800, 3), 100, np.uint8))
    return source


def test_incompatible_mask_requires_force_before_inference(tmp_path, monkeypatch):
    source = configure_outputs(tmp_path, monkeypatch)
    config.MASKS_DIR.mkdir(parents=True)
    path = config.MASKS_DIR/'test.png'
    cv2.imwrite(str(path), np.ones((10, 10), np.uint8)*255)
    before = path.read_bytes()
    monkeypatch.setattr(transfer, 'segment_details', lambda *a: pytest.fail('No inference during preflight'))
    with pytest.raises(ValueError, match='--force'):
        transfer.single(source)
    assert path.read_bytes() == before


def test_force_backs_up_old_mask_then_reuses_new_cache(tmp_path, monkeypatch):
    source = configure_outputs(tmp_path, monkeypatch)
    config.MASKS_DIR.mkdir(parents=True)
    mask_path = config.MASKS_DIR/'test.png'
    cv2.imwrite(str(mask_path), np.ones((10, 10), np.uint8)*255)
    before = mask_path.read_bytes()
    small_rgb = np.full((28, 100, 3), 100, np.uint8)
    small_mask = np.zeros((28, 100), bool)
    small_mask[10:, 20:80] = True
    info = dict(status='review', score=1.2, confidence=.8,
                prompts=[[50, 25]], labels=[1], candidates=[],
                **transfer.geometry((1000, 800, 3)))
    target_mask = transfer.project_mask(small_mask, (1000, 800, 3))
    monkeypatch.setattr(transfer, 'segment_details', lambda *a: (target_mask, small_mask, small_rgb, dict(info)))
    transfer.single(source, force=True)
    backups = list((config.OUTPUTS_DIR/'c_before_test_transfer').glob('*/test.png'))
    assert len(backups) == 1 and backups[0].read_bytes() == before
    monkeypatch.setattr(transfer, 'segment_details', lambda *a: pytest.fail('Matching cache should skip inference'))
    result = transfer.single(source)
    assert result[0]['status'] == 'review'
    assert 'test_c_review' in config.FLAGS_CSV.read_text()
    meta = json.loads(mask_path.with_suffix('.test_c.json').read_text())
    assert meta['mask_sha256'] == transfer._hash(mask_path)
    assert cv2.imread(str(mask_path), 0).shape == tuple(meta['target_shape'])


def test_cli_limit_and_force_forwarded(monkeypatch):
    calls = []
    monkeypatch.setattr(part_c, 'run', lambda *a, **kw: calls.append((a, kw)))
    part_c.main(['run', 'data/mini', '--limit', '3', '--force'])
    assert calls[0][1]['limit'] == 3 and calls[0][1]['force'] is True
