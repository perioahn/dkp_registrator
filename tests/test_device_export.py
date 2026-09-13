import io
import zipfile
import pytest
from PIL import Image
from test_workspace_api import client, add, engine_result, wait_done
from webapp import server as s


def results(client, monkeypatch):
    ids = add(client, 3)
    monkeypatch.setattr(s, 'register_test', engine_result)
    assert client.post('/api/register', json={}).status_code == 200
    wait_done()
    return ids


def test_download_exports_real_jpeg_and_zip_with_revision_guard(client, monkeypatch):
    fid, a, b = results(client, monkeypatch)
    versions = {mid:s.SESSION.display_result(mid)['id'] for mid in (a,b)}
    body = {'only':[a], 'expected_fixed_id':fid, 'expected_results':{a:versions[a]}}
    response = client.post('/api/export', json=body)
    assert response.status_code == 200, response.text
    assert response.headers['content-type'] == 'image/jpeg'
    assert Image.open(io.BytesIO(response.content)).format == 'JPEG'
    body.update(only=[a,b], expected_results=versions)
    response = client.post('/api/export', json=body)
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert len(archive.namelist()) == 2
        assert all(Image.open(io.BytesIO(archive.read(n))).format == 'JPEG' for n in archive.namelist())
    body['expected_results'] = {a:'stale',b:versions[b]}
    assert client.post('/api/export', json=body).status_code == 409


def test_device_switch_is_real_policy_and_preserves_images_masks(client, monkeypatch):
    import compute_device as device
    monkeypatch.setattr(device, 'accelerator', lambda:'cuda')
    monkeypatch.setattr(device, '_mode', 'auto')
    ids=add(client,2)
    before=s.SESSION.snapshot()
    s._sam_features['old-device'] = 'features'
    response=client.post('/api/gpu/device',json={'mode':'cpu'})
    assert response.status_code==200, response.text
    assert device.current()=='cpu'
    assert not s._sam_features
    assert s.SESSION.order==before['order']
    assert s.SESSION.masks==before['masks']
    assert client.post('/api/gpu/device',json={'mode':'auto'}).status_code==200
    assert device.current()=='cuda'
    s.SESSION.running=True
    assert client.post('/api/gpu/device',json={'mode':'cpu'}).status_code==409
    assert device.current()=='cuda'


def test_device_unavailable_and_inflight_mask_are_rejected(client, monkeypatch):
    import compute_device as device
    monkeypatch.setattr(device, 'accelerator',lambda:None)
    monkeypatch.setattr(device, '_mode','cpu')
    assert client.post('/api/gpu/device',json={'mode':'auto'}).status_code==422
    assert client.post('/api/gpu/device',json={'mode':'invalid'}).status_code==422
    with s._sam_lock:
        assert client.post('/api/gpu/device',json={'mode':'cpu'}).status_code==409


def test_folder_picker_cancel_and_failure_are_distinct(client, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:subprocess.CompletedProcess([],1,b'',b'failed'))
    assert client.post('/api/select_folder').status_code==500
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:subprocess.CompletedProcess([],0,b'',b''))
    assert client.post('/api/select_folder').json()=={'path':None}
