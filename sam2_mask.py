# -*- coding: utf-8 -*-
"""
sam2_mask.py — SAM2 interactive mask selection (reusable module).

조작법 (matplotlib 창):
  좌클릭:  치아 선택 (foreground)
  우클릭:  제외 영역 (background)
  x:       리셋 (전체 초기화)
  z:       현재 개체 마스크 확정 → 다음 개체
  c:       전체 완료 (확정된 마스크 union 반환)
  q:       취소
"""

import ssl, os, sys

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

import subprocess


def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])


try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Installing SAM-2 + huggingface_hub ...")
    pip_install("SAM-2", "huggingface_hub")
    from sam2.sam2_image_predictor import SAM2ImagePredictor

import cv2
import numpy as np
import torch

# ── SAM2 singleton ────────────────────────────────
_sam2_predictor = None


def load_sam2_predictor(model_name="facebook/sam2-hiera-large",
                        max_hole_area=300.0,
                        max_sprinkle_area=150.0):
    """SAM2 predictor 싱글턴 로딩."""
    global _sam2_predictor
    if _sam2_predictor is None:
        _sam2_predictor = SAM2ImagePredictor.from_pretrained(
            model_name,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
    return _sam2_predictor


# ── interactive mask selector (per-object) ───────
class MaskSelector:
    """개체별 순차 선택 → union 방식 마스크 셀렉터.

    각 치아를 개별적으로 선택(N키로 확정)한 뒤,
    F키로 전체 완료하면 모든 확정 마스크의 union을 반환.
    """

    def __init__(self, image_rgb, title, sam):
        self.image_rgb = image_rgb.copy()
        self.title = title
        self.sam = sam
        # 현재 작업 중인 개체
        self.fg = []
        self.bg = []
        self.current_mask = None
        # 확정된 개체 마스크 목록
        self.confirmed = []
        self.done = False
        self.cancelled = False

    def _on_click(self, event):
        if event.inaxes is None or self.done:
            return
        x, y = event.xdata, event.ydata
        if event.button == 1:
            self.fg.append([x, y])
            self._predict()
        elif event.button == 3:
            self.bg.append([x, y])
            self._predict()

    def _clean_single(self, mask_bool):
        """개별 개체 마스크 후처리: 큰 커널 + largest component only."""
        cleaned = clean_mask(mask_bool.astype(np.uint8),
                             close_kernel=11, open_kernel=5,
                             keep_largest=True)
        return cleaned.astype(bool)

    def _on_key(self, event):
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

    def _predict(self):
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

    def _redraw(self):
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

    def run(self):
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


def select_mask_interactive(image_rgb, title, predictor):
    """MaskSelector wrapper — returns mask as uint8 (0/1) or None."""
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

    def __init__(self, fixed_rgb, moving_rgb, sam):
        self.images = [fixed_rgb.copy(), moving_rgb.copy()]
        self.titles = ["Fixed", "Moving"]
        self.sam = sam
        self.active = 0
        self._last_set = -1
        # per-image state
        self.fg = [[], []]
        self.bg = [[], []]
        self.current_mask = [None, None]
        self.confirmed = [[], []]
        self.done = False
        self.cancelled = False

    def _clean_single(self, mask_bool):
        cleaned = clean_mask(mask_bool.astype(np.uint8),
                             close_kernel=11, open_kernel=5,
                             keep_largest=True)
        return cleaned.astype(bool)

    def _on_click(self, event):
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

    def _on_key(self, event):
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

    def _predict(self, i):
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

    def _redraw(self):
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

    def run(self):
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


def select_dual_mask_interactive(fixed_rgb, moving_rgb, predictor):
    """Fixed/Moving 병렬 마스크 선택 — 두 마스크 uint8 (0/1) 또는 None."""
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


# ── mask post-processing ─────────────────────────
def clean_mask(mask, close_kernel=7, open_kernel=3, min_area=50,
               keep_largest=False):
    """
    SAM2 마스크 형태학적 후처리.

    1. morphological close → 작은 구멍 채움
    2. morphological open → 작은 돌출/노이즈 제거
    3. component 필터링:
       - keep_largest=True: 최대 component만 유지 (개별 개체용)
       - keep_largest=False: min_area 이하 제거 (union 마스크용)
    4. contour 기반 내부 구멍 채움
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
            # 개별 개체: 최대 component만 유지
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            opened[labels != largest_label] = 0
        else:
            # union 마스크: min_area 이하만 제거
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
def resize_for_sam(img, max_side):
    """SAM2 작업용 리사이즈. 원본이 작으면 그대로."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h)), scale


def upscale_mask(mask, orig_h, orig_w):
    """SAM2 해상도 마스크를 원본 해상도로 업스케일."""
    if mask.shape[0] == orig_h and mask.shape[1] == orig_w:
        return mask
    upscaled = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST)
    return upscaled
