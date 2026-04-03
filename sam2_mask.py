# -*- coding: utf-8 -*-
"""SAM2 인터랙티브 마스크 선택 모듈.

조작법 (matplotlib 창):
    좌클릭: 치아 선택 (foreground)
    우클릭: 제외 영역 (background)
    x: 리셋 (전체 초기화)
    z: 현재 개체 마스크 확정 → 다음 개체
    c: 전체 완료 (확정된 마스크 union 반환)
    q: 취소
"""

from __future__ import annotations

import os
import ssl
import subprocess
import sys

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"


def _pip_install(*pkgs: str) -> None:
    """pip으로 패키지를 설치한다."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])


try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as exc:
    if getattr(sys, "frozen", False):
        raise RuntimeError(
            f"번들 내 모듈 로드 실패: {exc}. "
            "PyInstaller 빌드 시 --collect-all sam2 및 "
            "--collect-all torch 플래그를 확인하세요."
        ) from exc
    print("Installing sam2 + huggingface_hub ...")
    _pip_install("sam2", "huggingface_hub")
    from sam2.sam2_image_predictor import SAM2ImagePredictor

import cv2
import numpy as np
import torch

# ── SAM2 singleton ────────────────────────────────
_sam2_predictor: SAM2ImagePredictor | None = None


def load_sam2_predictor(
        model_name: str = "facebook/sam2-hiera-large",
        max_hole_area: float = 300.0,
        max_sprinkle_area: float = 150.0) -> SAM2ImagePredictor:
    """SAM2 predictor를 싱글턴으로 로딩한다.

    Args:
        model_name: HuggingFace 모델 이름.
        max_hole_area: 마스크 후처리 최대 hole 면적.
        max_sprinkle_area: 마스크 후처리 최대 sprinkle 면적.

    Returns:
        SAM2ImagePredictor 인스턴스.
    """
    global _sam2_predictor
    if _sam2_predictor is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _sam2_predictor = SAM2ImagePredictor.from_pretrained(
            model_name,
            device=device,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
    return _sam2_predictor


# ── interactive mask selector (per-object) ───────

class MaskSelector:
    """단일 이미지 인터랙티브 마스크 셀렉터.

    개체별로 순차 선택(Z키로 확정)한 뒤 C키로 전체 완료하면
    확정 마스크의 union을 반환한다.
    """

    def __init__(self, image_rgb: np.ndarray, title: str,
                 sam: SAM2ImagePredictor) -> None:
        self.image_rgb = image_rgb.copy()
        self.title = title
        self.sam = sam
        self.fg: list[list[float]] = []
        self.bg: list[list[float]] = []
        self.current_mask: np.ndarray | None = None
        self.confirmed: list[np.ndarray] = []
        self.done = False
        self.cancelled = False

    def _on_click(self, event) -> None:
        if event.inaxes is None or self.done:
            return
        x, y = event.xdata, event.ydata
        if event.button == 1:
            self.fg.append([x, y])
            self._predict()
        elif event.button == 3:
            self.bg.append([x, y])
            self._predict()

    def _clean_single(self, mask_bool: np.ndarray) -> np.ndarray:
        """개별 개체 마스크를 후처리한다.

        Args:
            mask_bool: bool 마스크.

        Returns:
            후처리된 bool 마스크.
        """
        cleaned = clean_mask(mask_bool.astype(np.uint8),
                             close_kernel=11, open_kernel=5,
                             keep_largest=True)
        return cleaned.astype(bool)

    def _on_key(self, event) -> None:
        if event.key == "z":
            # 현재 개체 확정 → 다음 개체
            if self.current_mask is not None:
                self.confirmed.append(self._clean_single(self.current_mask))
            self.fg.clear()
            self.bg.clear()
            self.current_mask = None
            # set_image 재호출 (SAM2 내부 상태 리셋)
            with torch.inference_mode():
                self.sam.set_image(self.image_rgb)
            self._redraw()
        elif event.key == "c":
            # 전체 완료: 현재 작업 중인 것도 포함
            if self.current_mask is not None:
                self.confirmed.append(self._clean_single(self.current_mask))
            self.done = True
            plt.close(self.fig)
        elif event.key == "x":
            # 리셋: 확정된 개체 포함 전체 초기화
            self.fg.clear()
            self.bg.clear()
            self.current_mask = None
            self.confirmed.clear()
            with torch.inference_mode():
                self.sam.set_image(self.image_rgb)
            self._redraw()
        elif event.key == "q":
            self.done = True
            self.cancelled = True
            plt.close(self.fig)

    def _predict(self) -> None:
        """현재 포인트로 SAM2 예측을 수행한다."""
        if not self.fg and not self.bg:
            return
        pts = np.array(self.fg + self.bg, dtype=np.float32)
        lbl = np.array([1] * len(self.fg) + [0] * len(self.bg), dtype=np.int32)
        with torch.inference_mode():
            masks, scores, _ = self.sam.predict(
                point_coords=pts, point_labels=lbl, multimask_output=True
            )
        best = np.argmax(scores)
        self.current_mask = masks[best].astype(bool)
        self._redraw()

    def _redraw(self) -> None:
        """마스크 오버레이와 포인트를 다시 그린다."""
        self.ax.clear()
        disp = self.image_rgb.copy().astype(float)

        # 확정된 마스크: 파란색 오버레이
        if self.confirmed:
            union_confirmed = np.zeros(disp.shape[:2], dtype=bool)
            for m in self.confirmed:
                union_confirmed |= m
            blue = np.zeros_like(self.image_rgb, dtype=float)
            blue[union_confirmed] = [60, 120, 255]
            disp[union_confirmed] = disp[union_confirmed] * 0.5 + blue[union_confirmed] * 0.5

        # 현재 작업 중 마스크: 녹색 오버레이
        if self.current_mask is not None:
            green = np.zeros_like(self.image_rgb, dtype=float)
            green[self.current_mask] = [0, 200, 0]
            disp[self.current_mask] = disp[self.current_mask] * 0.6 + green[self.current_mask] * 0.4
            # 현재 마스크 윤곽선
            cts, _ = cv2.findContours(
                self.current_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            for c in cts:
                c = c.squeeze()
                if c.ndim == 2 and len(c) > 1:
                    self.ax.plot(c[:, 0], c[:, 1], color="yellow", lw=1.5, alpha=0.9)

        self.ax.imshow(disp.astype(np.uint8))

        # 현재 포인트 표시
        for p in self.fg:
            self.ax.plot(p[0], p[1], "o", color="lime", ms=8, mec="white", mew=1.5)
        for p in self.bg:
            self.ax.plot(p[0], p[1], "o", color="red", ms=8, mec="white", mew=1.5)

        n_confirmed = len(self.confirmed)
        n_pts = len(self.fg) + len(self.bg)
        self.ax.set_title(
            f"{self.title}  [confirmed: {n_confirmed} | points: {n_pts}]\n"
            "L-click: select | R-click: exclude | X: reset | Z: next | C: finish",
            fontsize=10,
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def run(self) -> np.ndarray | None:
        """마스크 선택을 실행한다.

        Returns:
            확정된 마스크의 union (bool 배열) 또는 취소 시 None.
        """
        with torch.inference_mode():
            self.sam.set_image(self.image_rgb)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._redraw()
        plt.show()

        if self.cancelled or not self.confirmed:
            return None

        # union of all confirmed masks
        union = np.zeros(self.image_rgb.shape[:2], dtype=bool)
        for m in self.confirmed:
            union |= m
        return union


def select_mask_interactive(
        image_rgb: np.ndarray, title: str,
        predictor: SAM2ImagePredictor) -> np.ndarray | None:
    """단일 이미지 마스크 선택 래퍼.

    Args:
        image_rgb: RGB numpy 배열.
        title: 창 제목.
        predictor: SAM2ImagePredictor.

    Returns:
        마스크 uint8 (0/1) 또는 None.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    globals()["plt"] = plt

    mask = MaskSelector(image_rgb, title, predictor).run()
    if mask is not None:
        mask_uint8 = mask.astype(np.uint8)
        cleaned = clean_mask(mask_uint8)
        return (cleaned > 0).astype(np.uint8)
    return None


# ── dual (side-by-side) mask selector ────────────

class DualMaskSelector:
    """Fixed/Moving 병렬 마스크 선택.

    좌: Fixed, 우: Moving. 클릭한 쪽이 활성화(녹색 테두리).
    각 이미지에서 개별 개체를 Z로 확정, C로 전체 완료.
    """

    def __init__(self, fixed_rgb: np.ndarray, moving_rgb: np.ndarray,
                 sam: SAM2ImagePredictor) -> None:
        self.images = [fixed_rgb.copy(), moving_rgb.copy()]
        self.titles = ["Fixed", "Moving"]
        self.sam = sam
        self.active = 0
        self._last_set = -1
        # per-image state
        self.fg: list[list] = [[], []]
        self.bg: list[list] = [[], []]
        self.current_mask: list[np.ndarray | None] = [None, None]
        self.confirmed: list[list[np.ndarray]] = [[], []]
        self.done = False
        self.cancelled = False

    def _clean_single(self, mask_bool: np.ndarray) -> np.ndarray:
        """개별 개체 마스크를 후처리한다."""
        cleaned = clean_mask(mask_bool.astype(np.uint8),
                             close_kernel=11, open_kernel=5,
                             keep_largest=True)
        return cleaned.astype(bool)

    def _on_click(self, event) -> None:
        if event.inaxes is None or self.done:
            return
        if event.inaxes == self.axes[0]:
            i = 0
        elif event.inaxes == self.axes[1]:
            i = 1
        else:
            return
        self.active = i
        x, y = event.xdata, event.ydata
        if event.button == 1:
            self.fg[i].append([x, y])
            self._predict(i)
        elif event.button == 3:
            self.bg[i].append([x, y])
            self._predict(i)

    def _on_key(self, event) -> None:
        i = self.active
        if event.key == "z":
            if self.current_mask[i] is not None:
                self.confirmed[i].append(
                    self._clean_single(self.current_mask[i]))
            self.fg[i].clear()
            self.bg[i].clear()
            self.current_mask[i] = None
            with torch.inference_mode():
                self.sam.set_image(self.images[i])
            self._last_set = i
            self._redraw()
        elif event.key == "c":
            for j in range(2):
                if self.current_mask[j] is not None:
                    self.confirmed[j].append(
                        self._clean_single(self.current_mask[j]))
            self.done = True
            plt.close(self.fig)
        elif event.key == "x":
            self.fg[i].clear()
            self.bg[i].clear()
            self.current_mask[i] = None
            self.confirmed[i].clear()
            with torch.inference_mode():
                self.sam.set_image(self.images[i])
            self._last_set = i
            self._redraw()
        elif event.key == "q":
            self.done = True
            self.cancelled = True
            plt.close(self.fig)

    def _predict(self, i: int) -> None:
        """i번째 이미지에 대해 SAM2 예측을 수행한다."""
        if not self.fg[i] and not self.bg[i]:
            return
        if self._last_set != i:
            with torch.inference_mode():
                self.sam.set_image(self.images[i])
            self._last_set = i
        pts = np.array(self.fg[i] + self.bg[i], dtype=np.float32)
        lbl = np.array([1] * len(self.fg[i]) + [0] * len(self.bg[i]),
                        dtype=np.int32)
        with torch.inference_mode():
            masks, scores, _ = self.sam.predict(
                point_coords=pts, point_labels=lbl, multimask_output=True)
        best = np.argmax(scores)
        self.current_mask[i] = masks[best].astype(bool)
        self._redraw()

    def _redraw(self) -> None:
        """모든 패널의 마스크 오버레이와 포인트를 다시 그린다."""
        for i in range(2):
            ax = self.axes[i]
            ax.clear()
            disp = self.images[i].copy().astype(float)

            if self.confirmed[i]:
                union = np.zeros(disp.shape[:2], dtype=bool)
                for m in self.confirmed[i]:
                    union |= m
                blue = np.zeros_like(self.images[i], dtype=float)
                blue[union] = [60, 120, 255]
                disp[union] = disp[union] * 0.5 + blue[union] * 0.5

            if self.current_mask[i] is not None:
                green = np.zeros_like(self.images[i], dtype=float)
                green[self.current_mask[i]] = [0, 200, 0]
                disp[self.current_mask[i]] = (
                    disp[self.current_mask[i]] * 0.6
                    + green[self.current_mask[i]] * 0.4)
                cts, _ = cv2.findContours(
                    self.current_mask[i].astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cts:
                    c = c.squeeze()
                    if c.ndim == 2 and len(c) > 1:
                        ax.plot(c[:, 0], c[:, 1], color="yellow",
                                lw=1.5, alpha=0.9)

            ax.imshow(disp.astype(np.uint8))

            for p in self.fg[i]:
                ax.plot(p[0], p[1], "o", color="lime", ms=7,
                        mec="white", mew=1.2)
            for p in self.bg[i]:
                ax.plot(p[0], p[1], "o", color="red", ms=7,
                        mec="white", mew=1.2)

            nc = len(self.confirmed[i])
            np_ = len(self.fg[i]) + len(self.bg[i])
            title = f"{self.titles[i]}  [confirmed: {nc} | pts: {np_}]"

            if i == self.active:
                title = f"▶ {title}"
                for spine in ax.spines.values():
                    spine.set_edgecolor("#00FF00")
                    spine.set_linewidth(3)
                    spine.set_visible(True)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor("gray")
                    spine.set_linewidth(1)
                    spine.set_visible(True)

            ax.set_title(title, fontsize=10)
            ax.axis("off")

        self.fig.suptitle(
            "L-click: select | R-click: exclude | "
            "Z: next obj | X: reset side | C: finish | Q: cancel",
            fontsize=9, y=0.02)
        self.fig.canvas.draw_idle()

    def run(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """마스크 선택을 실행한다.

        Returns:
            (fixed_mask, moving_mask) 튜플. 각각 bool 배열 또는 None.
        """
        with torch.inference_mode():
            self.sam.set_image(self.images[0])
        self._last_set = 0
        self.fig, self.axes = plt.subplots(1, 2, figsize=(20, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._redraw()
        plt.show()

        if self.cancelled:
            return None, None

        results = []
        for i in range(2):
            if not self.confirmed[i]:
                results.append(None)
            else:
                union = np.zeros(self.images[i].shape[:2], dtype=bool)
                for m in self.confirmed[i]:
                    union |= m
                results.append(union)
        return results[0], results[1]


def select_dual_mask_interactive(
        fixed_rgb: np.ndarray, moving_rgb: np.ndarray,
        predictor: SAM2ImagePredictor,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fixed/Moving 병렬 마스크 선택 래퍼.

    Args:
        fixed_rgb: Fixed RGB numpy 배열.
        moving_rgb: Moving RGB numpy 배열.
        predictor: SAM2ImagePredictor.

    Returns:
        (fixed_mask, moving_mask) 튜플. 각각 uint8 (0/1) 또는 None.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    globals()["plt"] = plt

    f_mask, m_mask = DualMaskSelector(fixed_rgb, moving_rgb, predictor).run()

    out = []
    for mask in [f_mask, m_mask]:
        if mask is not None:
            cleaned = clean_mask(mask.astype(np.uint8))
            out.append((cleaned > 0).astype(np.uint8))
        else:
            out.append(None)
    return out[0], out[1]


# ── multi (N-image) mask selector ────────────────

class MultiMaskSelector:
    """N개 이미지 병렬 마스크 선택.

    클릭으로 이미지를 활성화(녹색 테두리).
    L-click: foreground, R-click: background, Z: confirm obj,
    X: reset active, C: finish all, Q: cancel.
    """

    _MAX_IMAGES = 12  # 그리드 최대 3x4

    def __init__(self, images: list[np.ndarray], titles: list[str],
                 sam: SAM2ImagePredictor) -> None:
        if len(images) > self._MAX_IMAGES:
            raise ValueError(
                f"최대 {self._MAX_IMAGES}개 이미지까지 지원합니다 "
                f"(입력: {len(images)}개)")
        self.n = len(images)
        self.images = [img.copy() for img in images]
        self.titles = list(titles)
        self.sam = sam
        self.active = 0
        self._last_set = -1
        self.fg: list[list[list[float]]] = [[] for _ in range(self.n)]
        self.bg: list[list[list[float]]] = [[] for _ in range(self.n)]
        self.current_mask: list[np.ndarray | None] = [None] * self.n
        self.confirmed: list[list[np.ndarray]] = [[] for _ in range(self.n)]
        self.done = False
        self.cancelled = False

    def _grid_layout(self) -> tuple[int, int]:
        """이미지 수에 따른 그리드 레이아웃을 반환한다.

        Returns:
            (rows, cols) 튜플.
        """
        n = self.n
        if n <= 2:
            return 1, n
        elif n <= 4:
            return 2, 2
        elif n <= 6:
            return 2, 3
        elif n <= 9:
            return 3, 3
        else:
            return 3, 4

    def _clean_single(self, mask_bool: np.ndarray) -> np.ndarray:
        """개별 개체 마스크를 후처리한다."""
        cleaned = clean_mask(mask_bool.astype(np.uint8),
                             close_kernel=11, open_kernel=5,
                             keep_largest=True)
        return cleaned.astype(bool)

    def _on_click(self, event) -> None:
        if event.inaxes is None or self.done:
            return
        for i in range(self.n):
            if event.inaxes == self.axes[i]:
                self.active = i
                x, y = event.xdata, event.ydata
                if event.button == 1:
                    self.fg[i].append([x, y])
                    self._predict(i)
                elif event.button == 3:
                    self.bg[i].append([x, y])
                    self._predict(i)
                return

    def _on_key(self, event) -> None:
        i = self.active
        if event.key == "z":
            if self.current_mask[i] is not None:
                self.confirmed[i].append(
                    self._clean_single(self.current_mask[i]))
            self.fg[i].clear()
            self.bg[i].clear()
            self.current_mask[i] = None
            with torch.inference_mode():
                self.sam.set_image(self.images[i])
            self._last_set = i
            self._redraw()
        elif event.key == "c":
            for j in range(self.n):
                if self.current_mask[j] is not None:
                    self.confirmed[j].append(
                        self._clean_single(self.current_mask[j]))
            self.done = True
            plt.close(self.fig)
        elif event.key == "x":
            self.fg[i].clear()
            self.bg[i].clear()
            self.current_mask[i] = None
            self.confirmed[i].clear()
            with torch.inference_mode():
                self.sam.set_image(self.images[i])
            self._last_set = i
            self._redraw()
        elif event.key == "q":
            self.done = True
            self.cancelled = True
            plt.close(self.fig)

    def _predict(self, i: int) -> None:
        """i번째 이미지에 대해 SAM2 예측을 수행한다."""
        if not self.fg[i] and not self.bg[i]:
            return
        if self._last_set != i:
            with torch.inference_mode():
                self.sam.set_image(self.images[i])
            self._last_set = i
        pts = np.array(self.fg[i] + self.bg[i], dtype=np.float32)
        lbl = np.array([1] * len(self.fg[i]) + [0] * len(self.bg[i]),
                        dtype=np.int32)
        with torch.inference_mode():
            masks, scores, _ = self.sam.predict(
                point_coords=pts, point_labels=lbl, multimask_output=True)
        best = np.argmax(scores)
        self.current_mask[i] = masks[best].astype(bool)
        self._redraw()

    def _redraw(self) -> None:
        """모든 패널의 마스크 오버레이와 포인트를 다시 그린다."""
        for i in range(self.n):
            ax = self.axes[i]
            ax.clear()
            disp = self.images[i].copy().astype(float)

            if self.confirmed[i]:
                union = np.zeros(disp.shape[:2], dtype=bool)
                for m in self.confirmed[i]:
                    union |= m
                blue = np.zeros_like(self.images[i], dtype=float)
                blue[union] = [60, 120, 255]
                disp[union] = disp[union] * 0.5 + blue[union] * 0.5

            if self.current_mask[i] is not None:
                green = np.zeros_like(self.images[i], dtype=float)
                green[self.current_mask[i]] = [0, 200, 0]
                disp[self.current_mask[i]] = (
                    disp[self.current_mask[i]] * 0.6
                    + green[self.current_mask[i]] * 0.4)
                cts, _ = cv2.findContours(
                    self.current_mask[i].astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cts:
                    c = c.squeeze()
                    if c.ndim == 2 and len(c) > 1:
                        ax.plot(c[:, 0], c[:, 1], color="yellow",
                                lw=1.5, alpha=0.9)

            ax.imshow(disp.astype(np.uint8))

            for p in self.fg[i]:
                ax.plot(p[0], p[1], "o", color="lime", ms=7,
                        mec="white", mew=1.2)
            for p in self.bg[i]:
                ax.plot(p[0], p[1], "o", color="red", ms=7,
                        mec="white", mew=1.2)

            nc = len(self.confirmed[i])
            np_ = len(self.fg[i]) + len(self.bg[i])
            title = f"{self.titles[i]}  [confirmed: {nc} | pts: {np_}]"

            if i == self.active:
                title = f"▶ {title}"
                for spine in ax.spines.values():
                    spine.set_edgecolor("#00FF00")
                    spine.set_linewidth(3)
                    spine.set_visible(True)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor("gray")
                    spine.set_linewidth(1)
                    spine.set_visible(True)

            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Hide unused axes
        rows, cols = self._grid_layout()
        for j in range(self.n, rows * cols):
            self.axes[j].set_visible(False)

        self.fig.suptitle(
            "L-click: select | R-click: exclude | "
            "Z: next obj | X: reset side | C: finish | Q: cancel",
            fontsize=9, y=0.02)
        self.fig.canvas.draw_idle()

    def run(self) -> list[np.ndarray | None]:
        """마스크 선택을 실행한다.

        Returns:
            N개 마스크의 리스트. 각각 bool 배열 또는 None.
        """
        with torch.inference_mode():
            self.sam.set_image(self.images[0])
        self._last_set = 0
        rows, cols = self._grid_layout()
        fig_w = min(7 * cols, 28)
        fig_h = min(6 * rows, 18)
        self.fig, axes_raw = plt.subplots(rows, cols,
                                          figsize=(fig_w, fig_h))
        # Flatten axes to list
        if rows == 1 and cols == 1:
            self.axes = [axes_raw]
        elif rows == 1 or cols == 1:
            self.axes = list(axes_raw)
        else:
            self.axes = [axes_raw[r][c]
                         for r in range(rows) for c in range(cols)]

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._redraw()
        plt.show()

        if self.cancelled:
            return [None] * self.n

        results = []
        for i in range(self.n):
            if not self.confirmed[i]:
                results.append(None)
            else:
                union = np.zeros(self.images[i].shape[:2], dtype=bool)
                for m in self.confirmed[i]:
                    union |= m
                results.append(union)
        return results


def select_multi_mask_interactive(
        images: list[np.ndarray], titles: list[str],
        predictor: SAM2ImagePredictor) -> list[np.ndarray | None]:
    """N개 이미지 병렬 마스크 선택 래퍼.

    Args:
        images: RGB numpy 배열 리스트.
        titles: 각 이미지의 제목 리스트.
        predictor: SAM2ImagePredictor.

    Returns:
        N개 마스크 (uint8 0/1) 또는 None 엔트리의 리스트.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    globals()["plt"] = plt

    masks = MultiMaskSelector(images, titles, predictor).run()

    out = []
    for mask in masks:
        if mask is not None:
            cleaned = clean_mask(mask.astype(np.uint8))
            out.append((cleaned > 0).astype(np.uint8))
        else:
            out.append(None)
    return out


# ── mask post-processing ─────────────────────────

def clean_mask(mask: np.ndarray, close_kernel: int = 7,
               open_kernel: int = 3, min_area: int = 50,
               keep_largest: bool = False) -> np.ndarray:
    """SAM2 마스크 형태학적 후처리.

    1. morphological close → 작은 구멍 채움
    2. morphological open → 작은 돌출/노이즈 제거
    3. component 필터링:
       - keep_largest=True: 최대 component만 유지 (개별 개체용)
       - keep_largest=False: min_area 이하 제거 (union 마스크용)
    4. contour 기반 내부 구멍 채움

    Args:
        mask: 입력 바이너리 마스크.
        close_kernel: close 연산 커널 크기.
        open_kernel: open 연산 커널 크기.
        min_area: 최소 component 면적.
        keep_largest: True이면 최대 component만 유지.

    Returns:
        후처리된 uint8 마스크 (0/255).
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    # 1. Close: 작은 구멍 채움 (dilate → erode)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kc)

    # 2. Open: 작은 노이즈/sprinkle 제거 (erode → dilate)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, ko)

    # 3. component 필터링
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if n_labels > 1:
        if keep_largest:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            opened[labels != largest_label] = 0
        else:
            for i in range(1, n_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    opened[labels == i] = 0

    # 4. contour 기반 내부 구멍 채움
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(opened)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)

    return filled


# ── utility functions ─────────────────────────────

def resize_for_sam(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """SAM2 작업용 리사이즈.

    Args:
        img: 입력 이미지.
        max_side: 최대 변 길이.

    Returns:
        (리사이즈된 이미지, 스케일 팩터) 튜플.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h)), scale


def upscale_mask(mask: np.ndarray, orig_h: int,
                 orig_w: int) -> np.ndarray:
    """SAM2 해상도 마스크를 원본 해상도로 업스케일한다.

    Args:
        mask: SAM2 해상도 마스크.
        orig_h: 원본 높이.
        orig_w: 원본 너비.

    Returns:
        원본 해상도 마스크.
    """
    if mask.shape[0] == orig_h and mask.shape[1] == orig_w:
        return mask
    upscaled = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST)
    return upscaled
