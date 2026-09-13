"""Isolated integration host. Deterministic inference only in this test process."""
import sys
import tempfile
from pathlib import Path
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root))
from webapp import server as s

scratch = tempfile.TemporaryDirectory(prefix="dkp-workspace-e2e-")
s.SESSION = s.Session(scratch.name)
s._torch_device = lambda: "cpu"

def mask(i):
    st = s._mask_state(i)
    st['current'] = infer(i, s.get_work(i), st['points'], None)

def infer(i, work, points, key):
    yy, xx = np.indices(work.shape[:2])
    result = np.zeros_like(xx, dtype=bool)
    for p in points:
        circle = (xx-p['x'])**2+(yy-p['y'])**2 < 70**2
        result = (result | circle) if p['label'] else (result & ~circle)
    return result

def register(f, m, *args, **kwargs):
    time.sleep(.65)
    registered = cv2.warpAffine(m, np.eye(2, 3), (f.shape[1], f.shape[0]))
    return [{'status':'pass','gate':'similarity','registered_img':registered,
             'false_color':s._display_false_color(f,registered),'match_viz':np.concatenate([f,registered],axis=1),
             'metrics':{'n_inlier':60,'inlier_ratio':.9,'reproj_median':.2},'M_full':np.eye(3)}]

s._predict_mask = mask
s._infer_mask = infer
s.register_test = register
s.register_test_lazy = register

def recommend(fixed, moving, fixed_masks, moving_masks):
    """Deterministic correspondences; production API and similarity fit stay real."""
    from anchor_recovery import select_suggestions
    time.sleep(.25)
    points = []
    for mask in fixed_masks:
        y, x = np.nonzero(mask)
        for yy in np.linspace(y.min()+8, y.max()-8, 6):
            for xx in np.linspace(x.min()+8, x.max()-8, 6):
                points.append([xx, yy])
    k0 = np.asarray(points)
    return select_suggestions(k0, k0 + [3, 2], np.full(len(points), .95), fixed_masks, moving_masks)

s.recommend_anchors = recommend
fixtures = Path(__file__).parent / 'fixtures'
fixtures.mkdir(exist_ok=True)
for i in range(1,9):
    img = Image.new('RGB',(1600,1000),(36+i*4,40,47))
    draw=ImageDraw.Draw(img)
    for j in range(8):
        x=180+j*158;y=300+int(90*np.sin(j/7*np.pi))
        draw.rounded_rectangle((x,y,x+135,y+215),30,fill=(206+i*3,203,184),outline=(130,126,117),width=5)
    draw.text((80,80),f'SYNTHETIC PHOTO {i}',fill='white',font_size=40)
    draw.rectangle((60,800,1540,805),fill=(50,155,120))
    img.save(fixtures/f'photo-{i}.png')

if __name__=='__main__':
    import uvicorn
    uvicorn.run(s.app,host='127.0.0.1',port=8792,log_level='warning')
