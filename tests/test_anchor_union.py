import numpy as np
import anchor_recovery as ar
from test_anchor_recovery import fixture_points
from test_workspace_api import client, add
from webapp import server as s


def test_same_union_gives_same_recommendations_despite_click_partition():
    masks, k0, k1, cf = fixture_points()
    union = np.logical_or.reduce(masks)
    pieces = []
    for left in range(0, union.shape[1], 30):
        piece = union.copy()
        piece[:, :left] = False
        piece[:, left+45:] = False
        pieces.append(piece)
    expected = ar.select_suggestions(k0, k1, cf, [union], [union])
    assert ar.select_suggestions(k0, k1, cf, pieces, pieces[::-1]) == expected
    assert ar.select_suggestions(k0, k1, cf, [union]*12, [union]*20) == expected
    pairs, _ = expected
    for left, right in ((20, 160), (290, 450)):
        region = [p for p in pairs if left <= p['fixed'][0] < right]
        assert len(region) >= 3
        assert np.ptp(np.array([p['fixed'] for p in region]), axis=0).min() > 50


def test_recommend_accepts_more_than_eight_selections(monkeypatch):
    import matching
    union = np.ones((160, 160), bool)
    points = np.array([(x, y) for x in (20, 60, 100, 140) for y in (20, 60, 100, 140)], float)
    calls = []
    def match(*args, **kwargs):
        calls.append(1)
        return points, points, np.full(len(points), .95)
    monkeypatch.setattr(matching, 'loftr_match', match)
    image = np.zeros((160, 160, 3), np.uint8)
    pairs, missing = ar.recommend_anchors(image, image, [union]*20, [union]*11)
    assert len(pairs) >= 8
    assert not missing
    assert len(calls) <= 5


def test_candidates_outside_either_union_are_excluded():
    mask = np.zeros((160, 160), bool)
    mask[10:150, 10:150] = True
    points = np.array([(x, y) for x in (20, 60, 100, 140) for y in (20, 60, 100, 140)], float)
    moving = points.copy()
    moving[0] = [155, 155]
    pairs, _ = ar.select_suggestions(points, moving, np.ones(len(points)), [mask], [mask])
    assert pairs
    assert all(mask[round(p['moving'][1]), round(p['moving'][0])] for p in pairs)


def test_api_unions_all_confirmed_and_current_masks_in_current_frame(client, monkeypatch):
    fid, mid = add(client, 2)
    originals = {}
    for iid in (fid, mid):
        mask = np.ones(s.get_work(iid).shape[:2], bool)
        parts = [s._freeze_mask(iid, mask) for _ in range(12)]
        s.SESSION.masks[iid] = {'points': [], 'confirmed': parts, 'current': mask.copy(), 'rev': 1}
        originals[iid] = parts
    projected = []
    original_project = s._project_mask
    def project(iid, mask):
        projected.append(iid)
        return original_project(iid, mask)
    monkeypatch.setattr(s, '_project_mask', project)
    def recommend(fixed, moving, fixed_masks, moving_masks):
        assert len(fixed_masks) == len(moving_masks) == 1
        assert fixed_masks[0].all() and moving_masks[0].all()
        return [], []
    monkeypatch.setattr(s, 'recommend_anchors', recommend)
    response = client.post(f'/api/anchors/{mid}/recommend', json={'fixed_id': fid})
    assert response.status_code == 200, response.text
    assert projected == [fid]*13 + [mid]*13
    for iid in (fid, mid):
        assert s.SESSION.masks[iid]['confirmed'] is originals[iid]
        assert len(originals[iid]) == 12


def test_four_sparse_correspondences_survive_a_dense_conflicting_area():
    mask = np.ones((240, 640), bool)
    sparse = np.array([[30, 30], [50, 30], [30, 60], [50, 60]], float)
    dense = np.array([(x, y) for x in range(350, 600, 10) for y in range(30, 200, 10)], float)
    k0 = np.vstack((sparse, dense))
    k1 = k0.copy(); k1[:4] += [0, 15]
    pairs, _ = ar.select_suggestions(k0, k1, np.ones(len(k0)), [mask], [mask])
    assert any(p['fixed'][0] < 100 for p in pairs)
