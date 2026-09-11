"""Scoring and prediction copied from Test/aibip/segmentation.py.

B-prepared RGB in; five independent seeds and 15 MobileSAM candidates, no new
postprocessing. The legacy crop API delegates resizing to B. Model
construction belongs to the target project's part_c.
"""
from types import SimpleNamespace
import cv2
import numpy as np

def vegetation(image):
    rgb = image.astype(np.float32) / 255
    r, g, b = np.moveaxis(rgb, -1, 0)
    return (g > 1.03 * r) & (g > 1.07 * b) & ((rgb.max(2) - rgb.min(2)) > 0.06)


def seed_component(mask, points):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    ids = {int(labels[min(int(y), mask.shape[0]-1), min(int(x), mask.shape[1]-1)]) for x, y in points}
    ids.discard(0)
    if not ids:
        return np.zeros_like(mask, dtype=bool)
    best = max(ids, key=lambda k: stats[k, cv2.CC_STAT_AREA])
    return labels == best


def candidate_score(mask, confidence, green, positives, texture=None, saturation=None):
    area = float(mask.mean())
    if area < 0.02 or area > 0.95:
        return -1e6, {'area': area, 'reason': 'implausible_area'}
    h, w = mask.shape
    cover = sum(bool(mask[min(int(y), h-1), min(int(x), w-1)]) for x, y in positives) / len(positives)
    bottom = float(mask[int(h*.75):, int(w*.3):int(w*.7)].mean())
    vegetation_fraction = float(green[mask].mean())
    bottom_any = float(mask[int(h*.85):].mean())
    texture_fraction = float(texture[mask].mean()) if texture is not None else 0.0
    saturation_fraction = float(saturation[mask].mean()) if saturation is not None else 0.0
    score = (float(confidence) + 0.65 * cover + 0.25 * bottom + 0.35 * bottom_any
             - 2.5 * vegetation_fraction - 1.2 * texture_fraction - 0.6 * saturation_fraction)
    return score, {'area': area, 'confidence': float(confidence), 'seed_coverage': cover,
                   'bottom_center': bottom, 'vegetation_fraction': vegetation_fraction,
                   'texture_fraction': texture_fraction, 'saturation_fraction': saturation_fraction}


class TestRoadSegmenter:
    def __init__(self, predictor, segment_width=1024):
        self.predictor = predictor
        self.config = SimpleNamespace(segment_width=segment_width)

    def predict(self, image):
        """Compatibility API for an RGB crop; all resizing belongs to B."""
        from aibip.part_b import resize_segmentation_rgb
        small = resize_segmentation_rgb(image, self.config.segment_width)
        return self.predict_preprocessed(small, image.shape[:2])

    def predict_preprocessed(self, small, crop_shape):
        """C consumes B's ready-to-use RGB pixels; prompts/scoring stay unchanged."""
        import torch
        h0, w0 = crop_shape
        h, w = small.shape[:2]
        green = vegetation(small)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        texture = cv2.Canny(gray, 60, 120).astype(np.float32) / 255
        saturation = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[:,:,1].astype(np.float32) / 255
        # Negative points must be green and lie outside the central road corridor.
        yy, xx = np.where(green & ((np.arange(w)[None, :] < .25*w) | (np.arange(w)[None, :] > .75*w)))
        negative = []
        for left in [True, False]:
            ids = np.flatnonzero(xx < w / 2 if left else xx >= w / 2)
            if len(ids):
                k = ids[np.argmin((yy[ids]-h*.65)**2 + (xx[ids]-w*(.15 if left else .85))**2)]
                negative.append((xx[k], yy[k]))
        candidates = []
        with torch.inference_mode():
            self.predictor.set_image(small)
            for cx in [.20, .35, .50, .65, .80]:
                # Separate seeds allow the road to move left/right at bends and junctions.
                # A mandatory centre seed farther up would incorrectly force a field mask.
                positives = [(cx*w, .92*h)]
                points = np.array(positives + negative, dtype=np.float32)
                labels = np.array([1]*len(positives) + [0]*len(negative))
                masks, confidences, _ = self.predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
                for mask, conf in zip(masks, confidences):
                    component = seed_component(mask, positives)
                    score, details = candidate_score(component, conf, green, positives, texture, saturation)
                    candidates.append((score, component, details, points, labels))
        candidates.sort(key=lambda c: c[0], reverse=True)
        score, mask, details, points, labels = candidates[0]
        if score < 0.5 or details.get('vegetation_fraction', 1) > .45:
            mask = np.zeros((h, w), bool)
            status = 'no_road'
        else:
            status = 'review' if details['confidence'] < .85 or details['vegetation_fraction'] > .12 else 'ok'
        info = {'status': status, 'score': score, **details,
                'prompts': points.tolist(), 'labels': labels.tolist(),
                'candidates': [{'score': c[0], **c[2]} for c in candidates]}
        full = cv2.resize(mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
        return full, mask, small, info
