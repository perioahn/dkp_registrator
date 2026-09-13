import io
import uuid
import numpy as np
from PIL import Image
from test_workspace_api import client, add
from webapp import server as s


def test_preview_cache_reuses_encoded_bytes_and_bounds_memory(client, monkeypatch):
    image_id = add(client, 1)[0]
    calls = []
    encode = s.cv2.imencode
    def counted(*args, **kwargs):
        calls.append(1)
        return encode(*args, **kwargs)
    monkeypatch.setattr(s.cv2, 'imencode', counted)
    for _ in range(3):
        client.get(f'/api/image/{image_id}').raise_for_status()
    assert len(calls) == 1
    monkeypatch.setattr(s, '_PREVIEW_BUDGET', 100)
    client.get(f'/api/image/{image_id}?max_side=20').raise_for_status()
    assert sum(map(len, s._preview_cache.values())) <= 100


def test_thumbnail_is_small_but_main_preview_is_unchanged(client):
    blob = io.BytesIO()
    Image.fromarray(np.zeros((1200, 1800, 3), np.uint8)).save(blob, 'JPEG')
    image_id = client.post('/api/upload', files=[('files', ('large.jpg', blob.getvalue(), 'image/jpeg'))]).json()['ids'][0]
    small = Image.open(io.BytesIO(client.get(f'/api/image/{image_id}?max_side=256').content))
    main = Image.open(io.BytesIO(client.get(f'/api/image/{image_id}').content))
    assert max(small.size) == 256
    assert max(main.size) == 1024


def test_mask_cache_reuses_but_undo_branch_cannot_show_old_pixels(client, monkeypatch):
    image_id = add(client, 1)[0]
    def predict(_):
        mask = np.zeros((80, 100), bool)
        x = int(s._mask_state(image_id)['points'][0]['x'])
        mask[10:30, x:x+10] = True
        s._mask_state(image_id)['current'] = mask
    monkeypatch.setattr(s, '_predict_mask', predict)
    def click(x):
        client.post(f'/api/mask/{image_id}/click', json={'x':x,'y':20,'label':1}).raise_for_status()
    click(10)
    old = client.get(f'/api/mask/{image_id}/overlay').content
    renderer = s._mask_overlay_png
    monkeypatch.setattr(s, '_mask_overlay_png', lambda *args: (_ for _ in ()).throw(AssertionError('repeat render')))
    assert client.get(f'/api/mask/{image_id}/overlay').content == old
    monkeypatch.setattr(s, '_mask_overlay_png', renderer)
    client.post('/api/history/undo').raise_for_status()
    click(60)
    assert client.get(f'/api/mask/{image_id}/overlay').content != old


def test_full_cache_budget_allows_oversized_uncached_image(client, monkeypatch):
    ids = add(client, 3)
    s._full_cache.clear()
    monkeypatch.setattr(s, '_FULL_BUDGET', 25000)
    for image_id in ids:
        assert s.get_full(image_id).shape == (80, 100, 3)
    assert sum(a.nbytes for a in s._full_cache.values()) <= 25000
    monkeypatch.setattr(s, '_FULL_BUDGET', 100)
    assert s.get_full(ids[0]).shape == (80, 100, 3)
    assert not s._full_cache


def test_false_color_regions_crop_before_compositing_and_keep_previous(client, monkeypatch):
    fixed_id, mid = add(client, 2)
    fixed = np.full((80, 100, 3), 30, np.uint8)
    registered = np.full_like(fixed, 150)
    previous = dict(id=uuid.uuid4().hex, fixed_img=fixed, registered_img=np.full_like(fixed, 70))
    result = dict(id=uuid.uuid4().hex, fixed_img=fixed, registered_img=registered, previous=previous)
    s.SESSION.result_pairs[fixed_id] = {mid:result}
    s.SESSION.displayed_results[mid] = fixed_id
    blend = s._display_false_color
    shapes = []
    def measured(a,b):
        shapes.append(a.shape)
        return blend(a,b)
    monkeypatch.setattr(s, '_display_false_color', measured)
    for prefix, image in [('', registered), ('/previous', previous['registered_img'])]:
        response = client.get(f'/api/result/{mid}{prefix}/region?kind=false_color&x=10&y=20&width=30&height=25')
        assert response.status_code == 200
        actual = np.array(Image.open(io.BytesIO(response.content)))
        assert np.array_equal(actual, blend(fixed[20:45,10:40], image[20:45,10:40]))
    assert shapes == [(25,30,3)] * 2


def test_false_color_dark_crop_matches_full_image():
    fixed = np.full((80,100,3), 100, np.uint8)
    moving = fixed.copy()
    fixed[10:20,10:20] = 1
    moving[10:20,10:20] = 1
    full = s._display_false_color(fixed, moving)
    crop = s._display_false_color(fixed[10:20,10:20], moving[10:20,10:20])
    assert np.array_equal(full[10:20,10:20], crop)
    assert crop.max() == 1


def test_result_cache_changes_with_result_id_and_preserves_previous(client, monkeypatch):
    fid, mid = add(client, 2)
    old = dict(id=uuid.uuid4().hex, registered_img=np.full((80,100,3), 70, np.uint8))
    s.SESSION.result_pairs[fid] = {mid:old}
    s.SESSION.displayed_results[mid] = fid
    first = client.get(f'/api/result/{mid}/registered').content
    encode = s.cv2.imencode
    monkeypatch.setattr(s.cv2, 'imencode', lambda *args: (_ for _ in ()).throw(AssertionError('repeat encoding')))
    assert client.get(f'/api/result/{mid}/registered').content == first
    monkeypatch.setattr(s.cv2, 'imencode', encode)
    s.SESSION.result_pairs[fid][mid] = dict(id=uuid.uuid4().hex, registered_img=np.full((80,100,3), 160, np.uint8), previous=old)
    assert client.get(f'/api/result/{mid}/registered').content != first
    assert client.get(f'/api/result/{mid}/previous/registered').content == first
