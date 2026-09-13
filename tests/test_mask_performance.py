"""Mask navigation must not scale with pixel work across the whole session."""
import numpy as np
import io
from pathlib import Path
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor
from test_workspace_api import client, add
from webapp import server as s


def test_state_does_not_decode_or_project_all_masked_photos(client, monkeypatch):
    ids = add(client, 7)
    for image_id in ids:
        st = s._mask_state(image_id)
        st['confirmed'] = [s._freeze_mask(image_id, np.ones((80, 100), bool))] * 5
    s._img_cache.clear()
    s._full_cache.clear()
    def unwanted(*args, **kwargs):
        raise AssertionError('State metadata must not decode photos or rasterize masks')
    monkeypatch.setattr(s, '_load_rgb', unwanted)
    monkeypatch.setattr(s, '_project_mask', unwanted)
    response = client.get('/api/state')
    assert response.status_code == 200
    photos = response.json()['images']
    assert len(photos) == 7
    assert all(p['mask_ready'] and p['n_objects'] == 5 for p in photos)
    assert all((p['w'], p['h']) == (100, 80) for p in photos)


def test_upload_retains_original_bytes_and_prepares_preview_once(client, monkeypatch):
    image = Image.fromarray(np.zeros((1200, 1800, 3), np.uint8))
    exif = Image.Exif()
    exif[274] = 6
    b = io.BytesIO()
    image.save(b, format='JPEG', exif=exif)
    contents = b.getvalue()
    response = client.post('/api/upload', files=[('files', ('oriented.jpg', contents, 'image/jpeg'))])
    image_id = response.json()['ids'][0]
    im = s.SESSION.images[image_id]
    assert Path(im['source_path']).read_bytes() == contents
    assert (im['full_w'], im['full_h']) == (1200, 1800)
    def unwanted(*args):
        raise AssertionError('Uploaded preview should already be available')
    monkeypatch.setattr(s, '_load_rgb', unwanted)
    assert client.get(f'/api/image/{image_id}').status_code == 200


def test_preview_does_not_hold_session_lock_and_discards_changed_session(client, monkeypatch):
    image_id = add(client, 1)[0]
    started, release = threading.Event(), threading.Event()
    def slow(*args):
        started.set()
        assert release.wait(5)
        return np.ones((80, 100), bool)
    monkeypatch.setattr(s, '_infer_mask', slow)
    with ThreadPoolExecutor() as executor:
        future = executor.submit(client.post, f'/api/mask/{image_id}/preview', json={'points':[{'x':20,'y':20,'label':1}]})
        try:
            assert started.wait(3)
            assert client.get('/api/state').json()['images'][0]['mask_ready'] is False
            assert client.get(f'/api/image/{image_id}').status_code == 200
            client.post('/api/reset').raise_for_status()
        finally:
            release.set()
        assert future.result().status_code == 409


def test_sam_image_features_reused_and_bounded_by_memory(client, monkeypatch):
    import sam2_mask
    class Predictor:
        device = 'cpu'
        _is_image_set = False
    predictor = Predictor()
    calls = []
    def encode(sam, work, **kwargs):
        calls.append(int(work[0,0,0]))
        sam._features = {'image_embed': np.full((8,8), work[0,0,0], np.float32)}
        sam._orig_hw = [work.shape[:2]]
        sam._is_image_set = True
    monkeypatch.setattr(s, '_get_sam', lambda: predictor)
    monkeypatch.setattr(sam2_mask, 'sam_set_image', encode)
    s._sam_features.clear()
    monkeypatch.setattr(s, '_sam_current', None)
    work = lambda i: np.full((10,10,3),i,np.uint8)
    s._sam_select('a', work(1), ('session','a',0))
    s._sam_select('b', work(2), ('session','b',0))
    s._sam_select('a', work(1), ('session','a',0))
    assert calls == [1,2]
    assert predictor._features['image_embed'][0,0] == 1
    for i in range(12):
        s._sam_select(str(i), work(i), ('session',str(i),0))
    assert len(s._sam_features) <= 4
    assert sum(entry[3] for entry in s._sam_features.values()) <= s._SAM_FEATURE_BUDGET
    monkeypatch.setattr(s, '_SAM_FEATURE_BUDGET', 300)
    s._sam_select('budget', work(1), ('session','budget',0))
    assert sum(entry[3] for entry in s._sam_features.values()) <= 300


def test_reset_releases_drafts_referencing_the_discarded_session(client):
    image_id = add(client, 1)[0]
    client.post(f'/api/mask/{image_id}/preview', json={'points':[{'x':20,'y':20,'label':1}]}).raise_for_status()
    assert s._mask_previews
    client.post('/api/reset').raise_for_status()
    assert not s._mask_previews
