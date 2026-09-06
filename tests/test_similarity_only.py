from dataclasses import replace
import numpy as np
import pytest
from config import DEFAULT, get_profile
from transform import is_similarity
from register import _run_gate


def test_similarity_validation_rejects_shear_and_nonuniform_scale():
    assert is_similarity(np.array([[0, -2, 4], [2, 0, 3], [0, 0, 1]]))
    assert not is_similarity(np.diag([2., 1., 1.]))
    assert not is_similarity(np.array([[1, .2, 0], [0, 1, 0], [0, 0, 1]]))
    assert not is_similarity(np.zeros((3, 3)))


def test_removed_profile_rejected():
    with pytest.raises(ValueError):
        get_profile("relaxed")


def test_affine_candidate_never_escapes_output(monkeypatch):
    import register as r
    pts = np.random.default_rng(2).uniform(0, 100, (60, 2)).astype(np.float32)
    inl = np.ones((60, 1), np.uint8)
    monkeypatch.setattr(r.cv2, "estimateAffinePartial2D", lambda *a, **k: (None, None))
    monkeypatch.setattr(r.cv2, "estimateAffine2D", lambda *a, **k: (np.array([[2, 0, 0], [0, 1, 0]], float), inl))
    M, _, gate, status, _, _ = _run_gate(pts, pts, np.ones(60), 10000, replace(DEFAULT, allow_affine=True))
    assert status in ("pass", "warn")
    assert gate == "similarity"
    assert is_similarity(M)


@pytest.mark.parametrize("fallback", [False, True])
def test_anchor_refit_scores_final_transform_and_keeps_similarity(monkeypatch, fallback):
    import register as r
    # Deliberately contradictory landmarks change the fit; old pre-fit zero metrics
    # would incorrectly pass this test.
    points = np.random.default_rng(3).uniform(10, 90, (50, 2)).astype(np.float32)
    monkeypatch.setattr(r, "auto_orient_and_crop", lambda image, mask: (image, mask, np.eye(3), (0, 0)))
    calls = []
    def match(*args, **kw):
        calls.append(1)
        if fallback and len(calls) == 1:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        return points.copy(), points.copy(), np.ones(50)
    monkeypatch.setattr(r, "_match_at_level", match)
    cfg = replace(DEFAULT, pyramid_levels=(100,))
    image = np.full((100, 100, 3), 127, np.uint8)
    mask = np.full((100, 100), 255, np.uint8)
    result = r.register_test(image, image, mask, mask,
                             anchor_points=[(35, 30, 30, 30), (75, 70, 70, 70)], cfg=cfg)[0]
    assert is_similarity(result["M_full"])
    expected = np.median(np.linalg.norm(points @ result["M_full"][:2, :2].T + result["M_full"][:2, 2] - points, axis=1))
    assert expected > 1
    assert result["metrics"]["reproj_median"] == pytest.approx(expected, abs=1e-6)
    assert len(result["metrics"]["anchor_residuals"]) == 2
    assert result["pyramid_level"] == (-1 if fallback else 0)
