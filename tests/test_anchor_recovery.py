"""Recovery contracts; synthetic points are not clinical validation."""
from collections import Counter
import numpy as np
import pytest

from anchor_recovery import select_suggestions, register_anchors
from transform import is_similarity
from test_workspace_api import client, add, wait_done, engine_result
from webapp import server as s


def fixture_points():
    f = np.zeros((240, 480), bool)
    a, b = f.copy(), f.copy()
    a[30:210, 20:160] = True
    b[30:210, 290:450] = True
    grid = np.array([(x, y) for x in range(300, 440, 10) for y in range(40, 200, 10)], float)
    sparse_region = np.array([(x, y) for x in (40, 100, 140) for y in (50, 120, 180)], float)
    k0 = np.vstack((grid, sparse_region))
    k1 = k0 - [4, 3]
    # Conflicting references must survive recommendation, not be discarded by global RANSAC.
    k1[-len(sparse_region):, 1] -= 14
    return [a, b], k0, k1, np.full(len(k0), .95)


def test_recommendations_cover_uneven_regions_without_projecting_existing_fit():
    masks, k0, k1, cf = fixture_points()
    pairs, missing = select_suggestions(k0, k1, cf, masks, masks)
    # Coverage now follows union geometry, not the original selection ids.
    assert len(pairs) <= 24
    for left, right in ((20, 160), (290, 450)):
        points = [p for p in pairs if left <= p['fixed'][0] < right]
        assert len(points) >= 3
        assert np.ptp(np.array([p['fixed'] for p in points]), axis=0).min() > 50
    for p in pairs:
        idx = np.where(np.all(k0 == p['fixed'], axis=1))[0][0]
        np.testing.assert_allclose(p['moving'], k1[idx])


def test_duplicate_density_and_low_confidence_cannot_create_support():
    masks, k0, k1, cf = fixture_points()
    only = k0[:, 0] > 200
    pairs, missing = select_suggestions(k0[only], k1[only], cf[only], masks, masks)
    assert 'fixed:0' in missing and 'moving:0' in missing
    assert pairs and all(p['fixed'][0] > 200 for p in pairs)
    p, _ = select_suggestions(np.tile(k0[:1], (100, 1)), np.tile(k1[:1], (100, 1)), np.ones(100), masks, masks)
    assert not p
    p, _ = select_suggestions(k0, k1, cf * .1, masks, masks)
    assert not p


def test_selected_region_order_and_location_do_not_change_coverage():
    masks, k0, k1, cf = fixture_points()
    expected = select_suggestions(k0, k1, cf, masks, masks)
    for regions in (masks, masks[::-1]):
        pairs, missing = select_suggestions(k0, k1, cf, regions, regions)
        assert (pairs, missing) == expected
    # Rotate the entire selection 180 degrees: no left/right or tooth-type priority.
    rotated_masks = [np.rot90(m, 2) for m in masks]
    pairs, missing = select_suggestions([479,239]-k0, [479,239]-k1, cf, rotated_masks, rotated_masks)
    assert len(pairs) <= 24
    for left, right in ((29, 189), (319, 459)):
        points = [p for p in pairs if left <= p['fixed'][0] <= right]
        assert len(points) >= 3
        assert np.ptp(np.array([p['fixed'] for p in points]), axis=0).min() > 50


def test_anchor_fit_works_without_automatic_matcher_and_preserves_ratio():
    image = np.zeros((240, 480, 3), np.uint8)
    moving = np.array([[30, 40], [150, 80], [270, 160]], float)
    theta = .13
    matrix = np.array([[1.1*np.cos(theta), -1.1*np.sin(theta), 7],
                       [1.1*np.sin(theta), 1.1*np.cos(theta), -2]])
    fixed = np.c_[moving, np.ones(3)] @ matrix.T
    result = register_anchors(image, image, np.c_[fixed, moving], ['manual']*3)
    assert is_similarity(result['M_full'])
    np.testing.assert_allclose(result['M_full'][:2], matrix, atol=1e-8)
    assert result['registered_img'].shape == image.shape
    assert result['metrics']['validation'] == 'fit_only'


def test_conflicting_groups_report_all_residuals_not_only_inliers():
    masks, k0, k1, cf = fixture_points()
    pairs, _ = select_suggestions(k0, k1, cf, masks, masks)
    image = np.zeros((240, 480, 3), np.uint8)
    result = register_anchors(image, image, [p['fixed']+p['moving'] for p in pairs], [p['group'] for p in pairs])
    assert result['status'] == 'warn'
    assert result['metrics']['reproj_p90'] > 2
    assert len(result['metrics']['anchor_residuals']) == len(pairs)
    assert {g['group'] for g in result['metrics']['reference_groups']} == {p['group'] for p in pairs}
    assert result['metrics']['reference_conflict']


@pytest.mark.parametrize('points', [[], [[1, 2, 3, 4]], [[1, 2, 3, 4]]*3,
                                  [[1, 2, 3, 4], [4, 6, float('nan'), 7]]])
def test_degenerate_anchors_rejected(points):
    image = np.zeros((20, 20, 3), np.uint8)
    with pytest.raises(ValueError):
        register_anchors(image, image, points, ['manual']*len(points))


def seed_regions(ids):
    for iid in ids:
        shape = s.get_work(iid).shape[:2]
        s.SESSION.masks[iid] = {'points': [], 'confirmed': [np.ones(shape, bool)], 'current': None, 'rev': 1}


def fake_recommend(*args):
    return [{'fixed': [10., 20.], 'moving': [12., 24.], 'group': '0:0'},
            {'fixed': [70., 60.], 'moving': [72., 64.], 'group': '0:0'}], []


def test_recommendation_is_nonmutating_cached_and_revision_bound(client, monkeypatch):
    fid, mid = add(client, 2)
    seed_regions([fid, mid])
    calls = []
    def recommend(*args):
        calls.append(1)
        return fake_recommend(*args)
    monkeypatch.setattr(s, 'recommend_anchors', recommend)
    before = s.SESSION.snapshot()
    revision = s.SESSION.revision
    a = client.post(f'/api/anchors/{mid}/recommend', json={'fixed_id': fid})
    assert a.status_code == 200, a.text
    draft = a.json()
    assert len(draft['pairs']) == 2
    assert s.SESSION.anchors[(fid, mid)]['pairs'] == []
    assert s.SESSION.result_pairs == before['result_pairs']
    assert s.SESSION.revision == revision
    assert client.post(f'/api/anchors/{mid}/recommend', json={'fixed_id': fid}).json() == draft
    assert len(calls) == 1
    s.SESSION.masks[mid]['rev'] += 1
    response = client.put(f'/api/anchors/{mid}', json={'fixed_id': fid, 'base_revision': 0,
        'pairs': draft['pairs'], 'input_token': draft['token']})
    assert response.status_code == 409
    assert s.SESSION.anchors[(fid, mid)]['pairs'] == []


def test_recommendation_rejects_changed_context_and_missing_masks(client, monkeypatch):
    fid, mid, other = add(client)
    monkeypatch.setattr(s, 'recommend_anchors', fake_recommend)
    assert client.post(f'/api/anchors/{mid}/recommend', json={'fixed_id': fid}).status_code == 422
    seed_regions([fid, mid])
    def changed(*args):
        s.SESSION.set_fixed(other)
        return fake_recommend(*args)
    monkeypatch.setattr(s, 'recommend_anchors', changed)
    assert client.post(f'/api/anchors/{mid}/recommend', json={'fixed_id': fid}).status_code == 409


def test_anchor_recovery_preserves_manual_anchors_previous_result_and_undo(client, monkeypatch):
    fid, mid = add(client, 2)
    seed_regions([fid, mid])
    monkeypatch.setattr(s, 'register_test', engine_result)
    client.post('/api/register', json={'only': [mid]}); wait_done()
    original = s.SESSION.display_result(mid)['id']
    monkeypatch.setattr(s, 'recommend_anchors', fake_recommend)
    manual = {'id': 'manual-1', 'fixed': [30, 40], 'moving': [32, 44], 'enabled': True}
    client.put(f'/api/anchors/{mid}', json={'fixed_id': fid, 'base_revision': 0, 'pairs': [manual]})
    draft = client.post(f'/api/anchors/{mid}/recommend', json={'fixed_id': fid}).json()
    saved = client.put(f'/api/anchors/{mid}', json={'fixed_id': fid, 'base_revision': 1,
        'pairs': [manual] + draft['pairs'], 'input_token': draft['token']})
    assert saved.status_code == 200, saved.text
    assert saved.json()['pairs'][0]['id'] == 'manual-1'
    def no_global(*args, **kwargs):
        raise AssertionError('anchor recovery must not invoke the global matcher')
    monkeypatch.setattr(s, 'register_test', no_global)
    response = client.post('/api/register', json={'only': [mid], 'anchor_only': True,
        'expected_fixed': fid, 'expected_anchor_revision': saved.json()['revision']})
    assert response.status_code == 200, response.text
    wait_done()
    current = s.SESSION.display_result(mid)
    assert current['gate'] == 'anchor_similarity'
    assert current['previous']['id'] == original
    assert is_similarity(current['M_full'])
    np.testing.assert_allclose(current['M_full'][:2, 2], [-2, -4], atol=1e-8)
    assert client.post(f'/api/result/{mid}/restore-previous', json={'result_id': 'old'}).status_code == 409
    restored = client.post(f'/api/result/{mid}/restore-previous', json={'result_id': current['id']})
    assert restored.status_code == 200 and restored.json()['id'] == original
    client.post('/api/history/undo')
    assert s.SESSION.display_result(mid)['id'] == current['id']


def test_recommendation_coordinates_return_to_original_frame(client, monkeypatch):
    fid, mid = add(client, 2)
    seed_regions([fid, mid])
    for iid in (fid, mid):
        s.SESSION.images[iid]['G'] = np.array([[2., 0, 4], [0, 2., 6], [0, 0, 1]])
    monkeypatch.setattr(s, 'recommend_anchors', fake_recommend)
    draft = client.post(f'/api/anchors/{mid}/recommend', json={'fixed_id': fid}).json()
    np.testing.assert_allclose(draft['pairs'][0]['fixed'], [3, 7])
    np.testing.assert_allclose(draft['pairs'][0]['moving'], [4, 9])


def test_invalid_anchor_retry_keeps_previous_result_and_rejects_stale_revision(client, monkeypatch):
    fid, mid = add(client, 2)
    monkeypatch.setattr(s, 'register_test', engine_result)
    client.post('/api/register', json={'only': [mid]}); wait_done()
    original = s.SESSION.display_result(mid)['id']
    points = [{'id': str(i), 'fixed': [20,20], 'moving': [30+i*10,30], 'enabled': True} for i in range(2)]
    client.put(f'/api/anchors/{mid}', json={'fixed_id': fid, 'base_revision': 0, 'pairs': points})
    data = {'only': [mid], 'anchor_only': True, 'expected_fixed': fid, 'expected_anchor_revision': 0}
    assert client.post('/api/register', json=data).status_code == 409
    data['expected_anchor_revision'] = 1
    assert client.post('/api/register', json=data).status_code == 200
    wait_done()
    kept = s.SESSION.display_result(mid)
    assert kept['id'] == original and kept['latest_attempt_failed']
