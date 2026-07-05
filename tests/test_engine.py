"""엔진 테스트: 설정/게이트/orientation 수학 + 합성 이미지 E2E (LoFTR 실사용).

실행: py -3.13 -m pytest tests -q  (LoFTR 가중치 첫 실행 시 자동 다운로드)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from config import DEFAULT, PROFILES, get_profile
from register import (
    _apply_orientation,
    _rescale_M,
    _to_3x3,
    _transform_anchors_orient,
    register_test,
    register_test_lazy,
)
from transform import quality_gate_similarity


# ── 순수 로직 ──────────────────────────────────────

def test_profiles():
    assert get_profile("strict").sim_gate.min_inlier_fail > \
        DEFAULT.sim_gate.min_inlier_fail
    assert get_profile("relaxed").sim_gate.rotation_fail_deg > \
        DEFAULT.sim_gate.rotation_fail_deg
    assert get_profile("없는이름") is DEFAULT
    assert set(PROFILES) == {"normal", "strict", "relaxed"}


def _perfect_match_set(n=60, rot_deg=5.0):
    """완벽히 대응하는 키포인트 셋 (similarity 변환)."""
    rng = np.random.default_rng(0)
    kpts_m = rng.uniform(50, 400, size=(n, 2)).astype(np.float32)
    th = np.radians(rot_deg)
    M = np.array([[np.cos(th), -np.sin(th), 10],
                  [np.sin(th), np.cos(th), -5]], dtype=np.float64)
    kpts_f = (M[:, :2] @ kpts_m.T).T + M[:, 2]
    inliers = np.ones((n, 1), dtype=np.uint8)
    return kpts_f.astype(np.float32), kpts_m, M, inliers


def test_gate_profiles_change_verdict():
    kf, km, M, inl = _perfect_match_set(rot_deg=17.0)  # normal fail(>20?) no: 17<20 → warn(>15)
    hull = cv2.contourArea(cv2.convexHull(kf))
    area = hull / 0.5  # coverage 0.5
    s_normal, _ = quality_gate_similarity(kf, km, M, inl, area)
    s_relaxed, _ = quality_gate_similarity(kf, km, M, inl, area,
                                           cfg=get_profile("relaxed").sim_gate)
    s_strict, _ = quality_gate_similarity(kf, km, M, inl, area,
                                          cfg=get_profile("strict").sim_gate)
    assert s_normal == 'warn'      # 회전 17° = normal warn 구간
    assert s_relaxed == 'pass'     # relaxed는 25°까지 허용
    assert s_strict == 'fail'      # strict는 15° 초과 시 fail


def test_orientation_roundtrip():
    img = np.arange(24, dtype=np.uint8).reshape(4, 6)
    for flip in (False, True):
        for k in range(4):
            out = _apply_orientation(img, flip, k)
            # 역변환: 회전 4-k + flip
            back = np.rot90(out, 4 - k) if k else out
            if flip:
                back = cv2.flip(np.ascontiguousarray(back), 1)
            assert np.array_equal(back, img), (flip, k)


def test_anchor_orientation_matches_image():
    """앵커 변환이 이미지 픽셀 변환과 일치하는지 — 마커 픽셀로 검증."""
    h, w = 40, 60
    img = np.zeros((h, w), dtype=np.uint8)
    mx, my = 13, 29
    img[my, mx] = 255
    for flip in (False, True):
        for k in range(4):
            out = _apply_orientation(img, flip, k)
            anchors = _transform_anchors_orient([(0, 0, mx, my)], w, h, flip, k)
            _, _, ax, ay = anchors[0]
            assert out[int(ay), int(ax)] == 255, (flip, k)


def test_rescale_M_identity():
    M = _to_3x3(np.array([[1.0, 0, 7], [0, 1.0, -3]]))
    out = _rescale_M(M, 0.5, 0.5, 1.0, 1.0)
    # fixed×2, moving×2 → 이동량도 2배
    assert np.allclose(out[:2, 2], [14, -6])
    assert np.allclose(out[:2, :2], M[:2, :2])


# ── 합성 이미지 E2E (LoFTR 실사용 — 수 초) ─────────

def _textured_image(seed=1, size=(480, 640)):
    """LoFTR가 매칭할 수 있는 풍부한 질감의 합성 이미지."""
    rng = np.random.default_rng(seed)
    img = np.full((*size, 3), 200, dtype=np.uint8)
    for _ in range(120):
        c = tuple(int(x) for x in rng.integers(30, 220, 3))
        p = (int(rng.integers(20, size[1] - 20)), int(rng.integers(20, size[0] - 20)))
        r = int(rng.integers(4, 22))
        cv2.circle(img, p, r, c, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def _warp_similarity(img, deg, scale, tx, ty):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h), borderValue=(200, 200, 200)), M


@pytest.mark.slow
def test_register_recovers_known_transform():
    fixed = _textured_image()
    moving, M_true = _warp_similarity(fixed, deg=-8.0, scale=1.05, tx=15, ty=-10)
    mask = np.full(fixed.shape[:2], 255, dtype=np.uint8)

    results = register_test(fixed, moving, mask, mask)
    r = results[0]
    assert r['status'] in ('pass', 'warn'), r.get('reason')
    reg = r['registered_img']
    # 정합 결과가 fixed와 픽셀 수준으로 유사해야 함 (경계 제외 중앙부)
    a = cv2.cvtColor(fixed[100:380, 150:490], cv2.COLOR_RGB2GRAY).astype(float)
    b = cv2.cvtColor(reg[100:380, 150:490], cv2.COLOR_RGB2GRAY).astype(float)
    mad = float(np.mean(np.abs(a - b)))
    assert mad < 12.0, f"mean abs diff {mad}"


@pytest.mark.slow
def test_lazy_prescreen_early_stop(monkeypatch):
    """프리스크리닝이 올바른 orientation을 상위로 올려 조기 종료하는지."""
    fixed = _textured_image(seed=2)
    # moving = fixed를 180° 뒤집은 것 (Lazy가 R180을 찾아야 함)
    moving = np.ascontiguousarray(np.rot90(fixed, 2))
    mask = np.full(fixed.shape[:2], 255, dtype=np.uint8)

    calls = []
    import register as reg_mod
    orig = reg_mod.register_test

    def counting(*a, **kw):
        calls.append(1)
        return orig(*a, **kw)

    monkeypatch.setattr(reg_mod, "register_test", counting)
    results = reg_mod.register_test_lazy(fixed, moving, mask, mask)
    r = results[0]
    assert r['status'] in ('pass', 'warn')
    assert r['lazy_orientation'] == (False, 2), r.get('lazy_label')
    # 8조합 전수 실행이 아니라 조기 종료했는지
    assert len(calls) <= DEFAULT.lazy_top_k, f"full runs: {len(calls)}"
