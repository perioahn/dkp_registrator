import io
import numpy as np
from PIL import Image
from test_workspace_api import client
from webapp import server as s


def test_editor_preview_is_bounded_cached_and_source_is_original_bytes(client,monkeypatch):
    image=Image.fromarray(np.full((4000,6000,3),155,np.uint8))
    data=io.BytesIO();image.save(data,'JPEG',quality=94)
    original=data.getvalue()
    image_id=client.post('/api/upload',files=[('files',('large.jpg',original,'image/jpeg'))]).json()['ids'][0]
    def no_full_decode(*args):
        raise AssertionError('editor opening must not use the full numpy decoder')
    monkeypatch.setattr(s,'_load_rgb',no_full_decode)
    response=client.get(f'/api/image/{image_id}/source-preview')
    assert response.status_code==200
    preview=Image.open(io.BytesIO(response.content))
    assert max(preview.size)<=1600
    assert response.headers['x-source-width']=='6000'
    assert response.headers['x-source-height']=='4000'
    with monkeypatch.context() as m:
        m.setattr(Image,'open',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('cached preview decoded again')))
        assert client.get(f'/api/image/{image_id}/source-preview').content==response.content
        source=client.get(f'/api/image/{image_id}/source')
        assert source.content==original and source.headers['content-type']=='image/jpeg'


def test_source_preview_respects_exif_orientation(client):
    image=Image.new('RGB',(80,40),'red')
    exif=Image.Exif();exif[274]=6
    data=io.BytesIO();image.save(data,'JPEG',exif=exif)
    image_id=client.post('/api/upload',files=[('files',('portrait.jpg',data.getvalue(),'image/jpeg'))]).json()['ids'][0]
    r=client.get(f'/api/image/{image_id}/source-preview')
    assert r.status_code==200
    assert Image.open(io.BytesIO(r.content)).size==(40,80)
    assert (r.headers['x-source-width'],r.headers['x-source-height'])==('40','80')


def test_import_converts_only_reduced_pixels_but_keeps_source_dimensions(client,monkeypatch):
    original=Image.new('RGB',(6000,4000),'#937e65')
    data=io.BytesIO();original.save(data,'JPEG')
    sizes=[]
    convert=Image.Image.convert
    def counted(image,*args,**kwargs):
        sizes.append(image.size)
        return convert(image,*args,**kwargs)
    monkeypatch.setattr(Image.Image,'convert',counted)
    image_id=client.post('/api/upload',files=[('files',('large.jpg',data.getvalue(),'image/jpeg'))]).json()['ids'][0]
    assert sizes and all(max(size)<=1024 for size in sizes)
    im=s.SESSION.images[image_id]
    assert (im['source_w'],im['source_h'],im['full_w'],im['full_h'])==(6000,4000,6000,4000)
    assert max(s.get_work(image_id).shape[:2])<=1024
