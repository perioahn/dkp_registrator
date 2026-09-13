from dataclasses import replace

import cv2
import numpy as np
import pytest

import register as r
import regional_matching as region
from config import DEFAULT
from transform import is_similarity
from test_workspace_api import client, add, wait_done, engine_result
from webapp import server as s


def grid():
    return np.array([(x, y) for y in range(30, 620, 35)
                     for x in range(30, 620, 35)], dtype=np.float64)


def test_level_bank_undoes_prewarp_resize_crop_and_rotation():
    original0, original1 = grid()[:20], grid()[:20] + [19, -7]
    rot0 = np.eye(3); rot0[:2] = cv2.getRotationMatrix2D((200, 200), 12, 1)
    rot1 = np.eye(3); rot1[:2] = cv2.getRotationMatrix2D((200, 200), -7, 1)
    prewarp = np.array([[1.1, -.1, 9], [.1, 1.1, 6], [0, 0, 1]])
    k0 = (region.project(original0, rot0) - [8, 12]) * .3
    k1 = region.project((region.project(original1, rot1) - [3, 15]) * .7, prewarp)
    bank = []
    region.collect_level(bank, k0, k1, np.ones(20), .3, .7, prewarp,
                         rot0, (8, 12), rot1, (3, 15))
    np.testing.assert_allclose(bank[0][0], original0, atol=1e-8)
    np.testing.assert_allclose(bank[0][1], original1, atol=1e-8)


def test_pool_deduplicates_overlapping_crops_without_fixed_cell_quotas():
    pts = grid()
    repeated = np.tile(pts, (5, 1))
    p0, p1, cf = region.bounded_pool(repeated, repeated, np.ones(len(repeated)), (640,640), (640,640))
    expected = len(pts)
    assert len(p0) == expected
    concentrated = np.zeros_like(repeated) + [50, 50]
    q0, _, _ = region.bounded_pool(repeated, concentrated, np.ones(len(repeated)), (640,640), (640,640))
    assert len(q0) == 1  # ambiguous many-to-one endpoints cannot dominate
    bad = np.r_[pts, [[np.nan, 5], [-1, 4], [700, 0]]]
    assert len(region.bounded_pool(bad, bad, np.ones(len(bad)), (640,640), (640,640))[0]) == expected


def test_windows_cover_separated_parts_instead_of_confidence_cluster():
    pts = grid()
    cf = np.where(pts[:, 0] < 160, .99, .5)
    windows = region.propose_windows(pts, pts + 2, cf, (640,640), (640,640))
    chosen = []
    for _ in range(4):
        chosen.append(region.next_window(windows,chosen,pts,pts+2,cf))
    centers = np.array([w.center0 for w in chosen])
    assert np.ptp(centers[:, 0]) > 350 and np.ptp(centers[:, 1]) > 350


def test_original_crop_keeps_coordinate_map_and_padding():
    from sliding_windows import paired_window, crop_pair
    image = np.arange(901*1403, dtype=np.uint32).reshape(901,1403).astype(np.uint8)
    pts = np.array([[1230, 790], [1370, 875]])
    window = paired_window(pts.mean(axis=0),400,np.array([[1,0,0],[0,1,0]],float),image.shape,image.shape)
    patch,_,(scale,offset),_ = crop_pair(image,image,window,1,1,320,2)
    assert patch.shape[0] % 8 == 0 and patch.shape[1] % 8 == 0
    assert max(patch.shape) <= 320
    np.testing.assert_allclose(((pts-offset)*scale)/scale+offset, pts)
    assert (offset > 0).all()


def test_validation_rejects_local_regression_even_if_other_regions_improve():
    pts = grid()[::3]
    baseline = np.array([[1, 0, 0], [0, 1, 0]], float)
    good = np.array([[1, 0, -2], [0, 1, 0]], float)
    assert region.validate_candidate(good, baseline, pts, pts+[2,0], (640,640), 3)[0]
    assert not region.validate_candidate(baseline, baseline, pts, pts, (640,640), 3)[0]
    rotated = np.array([[1, -.04, 12], [.04, 1, -12]], float)
    assert not region.validate_candidate(rotated, baseline, pts, pts, (640,640), 3)[0]
    assert not region.validate_candidate(good, None, pts[:4], pts[:4]+[2,0], (640,640), 3)[0]


@pytest.mark.parametrize('mode', ['unmasked', 'masked', 'anchors', 'disabled'])
def test_default_routing_only_unmasked_without_explicit_anchors(monkeypatch, mode):
    image = np.full((640,640,3), 90, np.uint8)
    mask = np.full((640,640), 255, np.uint8)
    if mode == 'masked':
        mask[:20] = 0
    called = []
    def pyramid(*a, **kw):
        assert (kw['correspondences'] is not None) == (mode == 'unmasked')
        return [{'status': 'fail', 'metrics': {}}]
    monkeypatch.setattr(r, '_register_pyramid', pyramid)
    monkeypatch.setattr(region, 'refine_result', lambda *a, **kw: called.append(1) or a[2])
    r.register_test(image, image, mask, mask,
                    anchor_points=[(1,1,1,1)] if mode == 'anchors' else None,
                    cfg=replace(DEFAULT, unmasked_refinement=mode != 'disabled'))
    assert bool(called) == (mode == 'unmasked')


def test_global_only_skips_duplicate_mask_inference(monkeypatch):
    calls = []
    pts = grid()[:20]
    monkeypatch.setattr(r, 'loftr_match', lambda *a, **k: (calls.append(1) or pts, pts, np.ones(20)))
    image = np.zeros((640,640), np.uint8)
    full = np.full_like(image,255)
    r._match_at_level(image,image,full,full,.2)
    assert len(calls) == 1
    partial = full.copy(); partial[:10] = 0
    r._match_at_level(image,image,partial,partial,.2)
    assert len(calls) == 3  # positive control for the masked dual pass


def test_no_seed_and_budget_keep_baseline_without_inference():
    image = np.zeros((640,640,3), np.uint8)
    baseline = {'status':'warn', 'M_full':np.eye(3), 'registered_img':image, 'metrics':{}}
    def forbidden(*a, **k):
        raise AssertionError('unexpected inference')
    empty = region.refine_result(image,image,baseline,[],DEFAULT,match_fn=forbidden)
    assert empty['M_full'] is baseline['M_full']
    pts = grid()
    stopped = region.refine_result(image,image,baseline,[(pts,pts,np.ones(len(pts)))],
                                  replace(DEFAULT,regional_budget_seconds=0),match_fn=forbidden)
    assert stopped['metrics']['regional']['attempted_crops'] == 0
    assert stopped['registered_img'] is image


def test_crop_failure_preserves_global_result():
    image = np.zeros((640,640,3), np.uint8)
    baseline = {'status':'warn', 'M_full':np.eye(3), 'registered_img':image, 'metrics':{}}
    pts = grid()
    def failure(*a, **kw):
        raise RuntimeError('inference failure')
    result = region.refine_result(image,image,baseline,[(pts,pts,np.ones(len(pts)))],DEFAULT,match_fn=failure)
    assert result['registered_img'] is image
    assert result['metrics']['regional']['reason'] == 'local_inference_failed'


def test_regional_fit_only_returns_similarity():
    pts = grid()
    outcome = region._fit(pts,pts+[3,4],np.ones(len(pts)),(640,640),(640,640),DEFAULT)
    assert outcome is not None and is_similarity(outcome[0])
    np.testing.assert_allclose(outcome[0], [[1,0,-3],[0,1,-4]], atol=1e-5)


@pytest.mark.parametrize('bad_crops', [False, True])
def test_complete_refinement_adopts_improvement_but_rejects_false_local_matches(bad_crops):
    image = np.zeros((640,640,3), np.uint8)
    baseline = {'status':'warn', 'M_full':np.eye(3), 'registered_img':image, 'metrics':{}}
    if not bad_crops:
        baseline['M_full'][0,2] = 2
    pts = grid()
    calls = []
    def match(a,b,**kw):
        calls.append(a.shape)
        p = np.array([(x,y) for y in range(15,a.shape[0]-25,13)
                      for x in range(15,a.shape[1]-40,13)], float)
        return p, p + ([25,0] if bad_crops else [0,0]), np.ones(len(p))
    result = region.refine_result(image,image,baseline,[(pts,pts,np.ones(len(pts)))],
                                  replace(DEFAULT,regional_budget_seconds=60),match_fn=match)
    assert 1 <= len(calls) <= DEFAULT.regional_max_crops
    assert result['metrics']['regional']['adopted'] == (not bad_crops)
    assert is_similarity(result['M_full'])
    np.testing.assert_allclose(result['M_full'],np.eye(3),atol=1e-5)


@pytest.mark.parametrize('mask_count',[0,1,2])
def test_api_only_enables_automatic_search_when_neither_photo_has_selection(client,monkeypatch,mask_count):
    ids = add(client,2)
    selected = set(ids[:mask_count])
    monkeypatch.setattr(s,'_union_mask',lambda iid: np.ones(s.get_work(iid).shape[:2],np.uint8)*255
                        if iid in selected else None)
    configs = []
    def engine(f,m,*a,**kw):
        configs.append(kw['cfg'])
        return engine_result(f,m)
    monkeypatch.setattr(s,'register_test',engine)
    assert client.post('/api/register',json={'only':[ids[1]]}).status_code == 200
    wait_done()
    assert configs[0].unmasked_refinement == (mask_count == 0)
    assert not s.get_anchors(ids[1])['pairs']
