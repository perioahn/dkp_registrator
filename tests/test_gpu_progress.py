import zipfile
from types import SimpleNamespace
from test_workspace_api import client
from webapp import server as s
import gpu_setup


def test_install_reports_real_extraction_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv('DKP_TORCH_DIR', str(tmp_path / 'cuda'))
    monkeypatch.setattr(gpu_setup, '_wheel_url', lambda pkg: f'https://test/{pkg}.whl')
    def download(url, dest, callback):
        with zipfile.ZipFile(dest, 'w') as archive:
            archive.writestr('torch/example', b'1234567')
        callback(100, 100)
    monkeypatch.setattr(gpu_setup, '_download', download)
    events = []
    gpu_setup.install_cuda(events.append)
    for pkg in ('torch', 'torchvision'):
        assert any(e.get('pkg') == pkg and e['phase'] == 'extract' and e.get('done') == 7 and e.get('total') == 7 for e in events)
    assert gpu_setup.installed()


def test_restart_guards_and_schedules_only_once(client, monkeypatch):
    fake = SimpleNamespace(should_exit=False, force_exit=False)
    scheduled = []
    monkeypatch.setattr(s, '_web_server', fake)
    monkeypatch.setattr(s, '_restart_requested', False)
    monkeypatch.setattr(s, '_gpu_state', {'installing': True})
    monkeypatch.setattr(gpu_setup, 'installed', lambda: True)
    monkeypatch.setattr(s.threading, 'Timer', lambda delay, fn: SimpleNamespace(start=lambda: scheduled.append(fn)))
    assert client.post('/api/gpu/restart').status_code == 409
    s._gpu_state['installing'] = False
    assert client.post('/api/gpu/restart').status_code == 200
    assert client.post('/api/gpu/restart').status_code == 200
    assert len(scheduled) == 1
    scheduled[0]()
    assert fake.should_exit and fake.force_exit
