"""파이프라인 설정 — 흩어져 있던 임계값의 단일 출처.

UI에는 PROFILES(normal/strict/relaxed)만 노출한다. 개별 값 튜닝은 코드에서.
GUI 의존성 없음.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SimilarityGate:
    """Similarity 변환 품질 게이트 임계값."""
    min_inlier_fail: int = 12
    min_inlier_warn: int = 30
    reproj_median_fail: float = 5.0
    reproj_p90_fail: float = 12.0
    reproj_median_warn: float = 3.0
    scale_fail: tuple[float, float] = (0.7, 1.4)
    scale_warn: tuple[float, float] = (0.8, 1.2)
    rotation_fail_deg: float = 20.0
    rotation_warn_deg: float = 15.0
    coverage_warn: float = 0.2


@dataclass(frozen=True)
class AffineGate:
    """Affine 변환 품질 게이트 임계값 (scale/rotation 무제한)."""
    min_inlier_fail: int = 12
    reproj_median_fail: float = 5.0
    coverage_warn: float = 0.15


@dataclass(frozen=True)
class PipelineConfig:
    """정합 파이프라인 전체 설정."""
    pyramid_levels: tuple[int, ...] = (320, 480, 640)
    pyramid_conf: float = 0.2
    clahe_clip: float = 2.0
    mask_sigma: int = 5
    ransac_thresh: float = 3.0
    min_matches: int = 4
    # Lazy 모드: 저해상 프리스크리닝으로 8조합 실행 순서 결정 (pass 시 조기종료)
    lazy_prescreen_side: int = 320
    sim_gate: SimilarityGate = SimilarityGate()
    aff_gate: AffineGate = AffineGate()


# UI 노출용 프로필 3종 — 개별 임계값은 노출하지 않는다.
PROFILES: dict[str, PipelineConfig] = {
    "normal": PipelineConfig(),
    "strict": PipelineConfig(
        sim_gate=SimilarityGate(
            min_inlier_fail=20, min_inlier_warn=45,
            reproj_median_fail=3.5, reproj_p90_fail=8.0,
            reproj_median_warn=2.0,
            scale_fail=(0.8, 1.25), scale_warn=(0.9, 1.1),
            rotation_fail_deg=15.0, rotation_warn_deg=10.0,
            coverage_warn=0.3),
        aff_gate=AffineGate(min_inlier_fail=20, reproj_median_fail=3.5,
                            coverage_warn=0.25),
    ),
    "relaxed": PipelineConfig(
        sim_gate=SimilarityGate(
            min_inlier_fail=8, min_inlier_warn=20,
            reproj_median_fail=7.0, reproj_p90_fail=16.0,
            reproj_median_warn=4.5,
            scale_fail=(0.55, 1.8), scale_warn=(0.7, 1.4),
            rotation_fail_deg=30.0, rotation_warn_deg=25.0,
            coverage_warn=0.1),
        aff_gate=AffineGate(min_inlier_fail=8, reproj_median_fail=7.0,
                            coverage_warn=0.08),
    ),
}

DEFAULT = PROFILES["normal"]


def get_profile(name: str) -> PipelineConfig:
    return PROFILES.get(name, DEFAULT)
