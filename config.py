"""파이프라인 설정 — 흩어져 있던 임계값의 단일 출처.

UI에는 PROFILES(normal/strict)만 노출한다. 개별 값 튜닝은 코드에서.
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
    # False = similarity만 사용 (비등방 스케일·전단 금지 — 비율 보존).
    # 같은 악궁을 두 번 찍은 사진의 실제 관계는 similarity(회전+등방배율+평행이동)다.
    # affine은 전단·비등방 배율을 허용해 매칭이 나쁠 때 "억지로 맞춘" 왜곡 결과를 낸다
    # (실사고 2회). 아래 필드는 기존 호출 호환용이며 모든 출력 경로에서 무시한다.
    allow_affine: bool = False
    # Lazy 모드: 저해상 프리스크리닝으로 8조합 실행 순서 결정 (pass 시 조기종료)
    lazy_prescreen_side: int = 320
    sim_gate: SimilarityGate = SimilarityGate()
    aff_gate: AffineGate = AffineGate()


# UI 노출용 프로필 2종 — 개별 임계값은 노출하지 않는다.
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
}

DEFAULT = PROFILES["normal"]


def get_profile(name: str) -> PipelineConfig:
    if name not in PROFILES:
        raise ValueError(f"지원하지 않는 정합 프로필: {name}")
    return PROFILES[name]
