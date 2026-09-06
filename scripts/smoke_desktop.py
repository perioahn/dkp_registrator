"""Start the actual packaged executable and verify static UI and upload API."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
from PIL import Image, ImageDraw

binary = Path('dist/DKPregistrator/DKPregistrator.exe') if sys.platform == 'win32' else Path('dist/DKPregistrator.app/Contents/MacOS/DKPregistrator')
assert binary.is_file(), binary
port = 18798
base = f'http://127.0.0.1:{port}'
with tempfile.TemporaryDirectory(prefix='dkp-native-smoke-') as temp:
    env = dict(os.environ, DKP_SESSION_ROOT=temp)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    process = subprocess.Popen([str(binary.resolve()), '--no-browser', '--persist', '--port', str(port)], env=env, creationflags=flags)
    try:
        limit = time.monotonic() + 150
        while True:
            try:
                with urllib.request.urlopen(base+'/api/state', timeout=3) as response:
                    state = json.load(response)
                break
            except Exception:
                if process.poll() is not None or time.monotonic() > limit:
                    raise RuntimeError(f'Packaged server did not start; exit={process.poll()}')
                time.sleep(1)
        assert state['images'] == [] and not state['running']
        with urllib.request.urlopen(base+'/') as response:
            assert b'<script' in response.read()
        with urllib.request.urlopen(base+'/api/app') as response:
            identity=json.load(response)
        assert identity['version'] == '1.5.1'
        pixels = io.BytesIO()
        photo=Image.new('RGB',(256,256),(35,35,35))
        ImageDraw.Draw(photo).ellipse((60,40,190,220), fill=(240,230,200))
        photo.save(pixels,format='PNG')
        boundary='DKPsmokeBoundary'
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="smoke.png"\r\nContent-Type: image/png\r\n\r\n'.encode()+pixels.getvalue()+f'\r\n--{boundary}--\r\n'.encode())
        request=urllib.request.Request(base+'/api/upload',data=body,headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
        with urllib.request.urlopen(request) as response:
            uploaded=json.load(response)
        assert len(uploaded['ids']) == 1
        with urllib.request.urlopen(base+'/api/state') as response:
            state=json.load(response)
        assert state['fixed_id'] == uploaded['ids'][0]
        with urllib.request.urlopen(base+f"/api/image/{uploaded['ids'][0]}/source") as response:
            assert Image.open(io.BytesIO(response.read())).size == (256,256)
        image_id=uploaded['ids'][0]
        request=urllib.request.Request(base+f'/api/mask/{image_id}/click', data=json.dumps({'x':128,'y':128,'label':1}).encode(), headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                assert len(json.load(response)['points']) == 1
        except urllib.error.HTTPError as exc:
            raise RuntimeError(exc.read().decode()) from exc
        with urllib.request.urlopen(base+f'/api/mask/{image_id}/overlay') as response:
            overlay=Image.open(io.BytesIO(response.read()))
            assert overlay.getchannel('A').getextrema()[1] > 0
        print('REAL SAM MASK VERIFIED', flush=True)
        print('NATIVE EXECUTABLE VERIFIED')
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=10)
