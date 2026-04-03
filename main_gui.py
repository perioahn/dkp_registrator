# -*- coding: utf-8 -*-
"""치아 정합 파이프라인 GUI.

SAM2 마스크 선택 → Register (8-combo 비교) → 결과 선택/저장.
"""

from __future__ import annotations

import gc
import io
import os
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from register import (
    CLAHE_CLIPS,
    CONF_LEVELS,
    MASK_SIGMAS,
    MAX_SIDES,
    false_color,
    register_test,
)
from sam2_mask import (
    clean_mask,
    load_sam2_predictor,
    resize_for_sam,
    select_multi_mask_interactive,
    upscale_mask,
)

SAM2_MAX_SIDE = 1024
THUMB_MAX = 400


def load_image_rgb(path: str) -> np.ndarray:
    """이미지 파일을 RGB numpy 배열로 로드한다.

    Args:
        path: 이미지 파일 경로. 한글 경로 지원.

    Returns:
        RGB numpy 배열 (H, W, 3).

    Raises:
        FileNotFoundError: 이미지 디코딩 실패 시.
    """
    buf = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"이미지 로드 실패: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def np_to_photo(img_rgb: np.ndarray,
                max_side: int = THUMB_MAX) -> ImageTk.PhotoImage:
    """numpy RGB 배열을 썸네일 PhotoImage로 변환한다.

    Args:
        img_rgb: RGB numpy 배열.
        max_side: 썸네일 최대 변 길이.

    Returns:
        tkinter PhotoImage 객체.
    """
    h, w = img_rgb.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h))
    return ImageTk.PhotoImage(Image.fromarray(img_rgb))


class TextRedirector(io.TextIOBase):
    """stdout/stderr를 Text 위젯으로 리디렉트한다."""

    def __init__(self, text_widget: tk.Text) -> None:
        self.text = text_widget

    def write(self, s: str) -> int:
        if s:
            try:
                self.text.after(0, self._append, s)
            except RuntimeError:
                pass  # widget destroyed
        return len(s) if s else 0

    def _append(self, s: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", s)
        self.text.see("end")
        self.text.configure(state="disabled")

    def flush(self) -> None:
        pass


class MainGUI:
    """치아 정합 파이프라인 메인 GUI.

    Attributes:
        fixed_img: 고정상 원본 해상도 RGB 배열.
        fixed_mask: 고정상 마스크 uint8 (0/255).
        moving_imgs: 이동상 RGB 배열 리스트.
        moving_masks: 이동상 마스크 리스트.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("치아 정합 파이프라인")
        self.root.geometry("900x950")

        self.fixed_img: np.ndarray | None = None
        self.fixed_mask: np.ndarray | None = None
        self.result: dict | None = None
        self._photo_refs: list[ImageTk.PhotoImage] = []
        self._best_match_hint: tuple | None = None
        # Anchor points per moving: {moving_idx: [(fx, fy, mx, my), ...]}
        self.anchor_points_per_moving: dict[int, list[tuple]] = {}

        # Multi-moving state
        self.moving_imgs: list[np.ndarray | None] = [None]
        self.moving_masks: list[np.ndarray | None] = [None]
        self.moving_paths: list[str] = [""]
        self.moving_entries: list[ttk.Entry] = []
        self.moving_rows: list[tuple] = []

        # Multi-registration results
        self._multi_regtest_results: dict[int, list[dict]] = {}
        self._multi_regtest_selected: dict[int, int] = {}

        self._build_ui()
        self._redirect_console()

    def _build_ui(self) -> None:
        """GUI 위젯을 생성하고 배치한다."""
        # ── 파일 선택 ──
        file_frame = ttk.LabelFrame(self.root, text="이미지 선택", padding=5)
        file_frame.pack(fill="x", padx=5, pady=3)
        self.file_frame = file_frame

        ttk.Label(file_frame, text="Fixed:").grid(row=0, column=0, sticky="w")
        self.fixed_entry = ttk.Entry(file_frame, width=60)
        self.fixed_entry.grid(row=0, column=1, padx=3)
        ttk.Button(file_frame, text="Browse",
                   command=self._browse_fixed).grid(row=0, column=2)

        # Moving1 row
        self._add_moving_row(0)

        # +/- buttons
        btn_sub = ttk.Frame(file_frame)
        btn_sub.grid(row=100, column=0, columnspan=3, sticky="w", pady=2)
        self._add_btn = ttk.Button(btn_sub, text="+ Moving", width=10,
                                   command=self._add_moving_slot)
        self._add_btn.pack(side="left", padx=2)
        self._remove_btn = ttk.Button(btn_sub, text="- Moving", width=10,
                                      command=self._remove_moving_slot,
                                      state="disabled")
        self._remove_btn.pack(side="left", padx=2)

        # ── SAM2 마스크 ──
        mask_frame = ttk.LabelFrame(self.root, text="SAM2 마스크", padding=5)
        mask_frame.pack(fill="x", padx=5, pady=3)
        ttk.Button(mask_frame, text="Select Masks (SAM2)",
                   command=self._select_masks).pack(side="left")
        self.mask_label = ttk.Label(mask_frame, text="  마스크 미선택")
        self.mask_label.pack(side="left", padx=10)

        # ── 실행 버튼 ──
        btn_frame = ttk.Frame(self.root, padding=5)
        btn_frame.pack(fill="x", padx=5)
        self.register_btn = ttk.Button(btn_frame, text="Register",
                                       command=self._run_register)
        self.register_btn.pack(side="left")
        self.status_label = ttk.Label(btn_frame, text="")
        self.status_label.pack(side="left", padx=10)

        # ── 결과 표시 ──
        result_frame = ttk.LabelFrame(self.root, text="결과", padding=5)
        result_frame.pack(fill="x", padx=5, pady=3)
        self.img_frame = ttk.Frame(result_frame)
        self.img_frame.pack()
        self.registered_label = ttk.Label(self.img_frame)
        self.registered_label.pack(side="left", padx=5)
        self.falsecolor_label = ttk.Label(self.img_frame)
        self.falsecolor_label.pack(side="left", padx=5)
        self.metrics_text = tk.Text(result_frame, height=5, width=90,
                                    state="disabled", font=("Consolas", 9))
        self.metrics_text.pack(fill="x", pady=3)

        # ── 콘솔 ──
        console_frame = ttk.LabelFrame(self.root, text="콘솔", padding=5)
        console_frame.pack(fill="both", expand=True, padx=5, pady=3)
        self.console = tk.Text(console_frame, height=10, state="disabled",
                               font=("Consolas", 9), bg="#1e1e1e", fg="#cccccc")
        scroll = ttk.Scrollbar(console_frame, command=self.console.yview)
        self.console.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.console.pack(fill="both", expand=True)

    def _redirect_console(self) -> None:
        """stdout/stderr를 콘솔 위젯으로 리디렉트한다."""
        redir = TextRedirector(self.console)
        sys.stdout = redir
        sys.stderr = redir

    # ── 파일 브라우즈 ──

    def _browse_fixed(self) -> None:
        """고정상 이미지 파일을 선택하고 로드한다."""
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not path:
            return
        self.fixed_entry.delete(0, "end")
        self.fixed_entry.insert(0, path)
        try:
            img = load_image_rgb(path)
            self.fixed_img = img
            self.fixed_mask = None
            self._best_match_hint = None
            self.result = None
            self._multi_regtest_results = {}
            self._multi_regtest_selected = {}
            self.anchor_points_per_moving.clear()
            self._update_mask_label()
            self._update_anchor_label()
            print(f"[INFO] Fixed 로드: {os.path.basename(path)} "
                  f"({img.shape[1]}x{img.shape[0]})")
            self._show_fixed_preview()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _browse_moving(self, moving_idx: int) -> None:
        """이동상 이미지 파일을 선택하고 로드한다.

        Args:
            moving_idx: 이동상 슬롯 인덱스.
        """
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not path:
            return
        entry = self.moving_entries[moving_idx]
        entry.delete(0, "end")
        entry.insert(0, path)
        try:
            img = load_image_rgb(path)
            self.moving_imgs[moving_idx] = img
            self.moving_masks[moving_idx] = None
            self.moving_paths[moving_idx] = path
            self._best_match_hint = None
            self.result = None
            self._multi_regtest_results = {}
            self._multi_regtest_selected = {}
            self.anchor_points_per_moving.pop(moving_idx, None)
            self._update_mask_label()
            self._update_anchor_label()
            if self.fixed_img is not None:
                self._show_fixed_preview()
            print(f"[INFO] Moving{moving_idx + 1} 로드: "
                  f"{os.path.basename(path)} "
                  f"({img.shape[1]}x{img.shape[0]})")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ── Moving slot 관리 ──

    def _add_moving_row(self, moving_idx: int) -> None:
        """파일 선택 프레임에 이동상 행을 추가한다.

        Args:
            moving_idx: 이동상 슬롯 인덱스.
        """
        row = moving_idx + 1
        lbl = ttk.Label(self.file_frame, text=f"Moving{moving_idx + 1}:")
        lbl.grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(self.file_frame, width=60)
        entry.grid(row=row, column=1, padx=3)
        btn = ttk.Button(self.file_frame, text="Browse",
                         command=lambda idx=moving_idx: self._browse_moving(idx))
        btn.grid(row=row, column=2)
        self.moving_entries.append(entry)
        self.moving_rows.append((lbl, entry, btn))

    _MAX_MOVING = 11  # MultiMaskSelector 그리드 최대 3x4=12, fixed 포함

    def _add_moving_slot(self) -> None:
        """이동상 슬롯을 추가하고 파일 선택 대화상자를 연다."""
        if len(self.moving_imgs) >= self._MAX_MOVING:
            messagebox.showinfo("Info", f"최대 {self._MAX_MOVING}개까지 추가 가능합니다.")
            return
        idx = len(self.moving_imgs)
        self.moving_imgs.append(None)
        self.moving_masks.append(None)
        self.moving_paths.append("")
        self._add_moving_row(idx)
        self._remove_btn.config(state="normal")
        self._browse_moving(idx)

    def _remove_moving_slot(self) -> None:
        """마지막 이동상 슬롯을 제거한다."""
        if len(self.moving_imgs) <= 1:
            return
        idx = len(self.moving_imgs) - 1
        lbl, entry, btn = self.moving_rows.pop()
        lbl.destroy()
        entry.destroy()
        btn.destroy()
        self.moving_entries.pop()
        self.moving_imgs.pop()
        self.moving_masks.pop()
        self.moving_paths.pop()
        self._multi_regtest_results.pop(idx, None)
        self._multi_regtest_selected.pop(idx, None)
        self._add_btn.config(state="normal")
        if len(self.moving_imgs) <= 1:
            self._remove_btn.config(state="disabled")

    def _compact_moving_slots(self, loaded_indices: list[int]) -> None:
        """빈 이동상 슬롯을 제거하고 번호를 재정렬한다.

        Args:
            loaded_indices: 이미지가 로드된 슬롯의 인덱스 리스트.
        """
        new_imgs = [self.moving_imgs[i] for i in loaded_indices]
        new_masks = [self.moving_masks[i] for i in loaded_indices]
        new_paths = [self.moving_paths[i] for i in loaded_indices]
        # Remove all moving rows
        while self.moving_rows:
            lbl, entry, btn = self.moving_rows.pop()
            lbl.destroy()
            entry.destroy()
            btn.destroy()
        self.moving_entries.clear()
        self.moving_imgs = new_imgs
        self.moving_masks = new_masks
        self.moving_paths = new_paths
        self._multi_regtest_results.clear()
        self._multi_regtest_selected.clear()
        # Re-add rows
        for idx in range(len(self.moving_imgs)):
            self._add_moving_row(idx)
            if self.moving_paths[idx]:
                self.moving_entries[idx].insert(0, self.moving_paths[idx])
        self._remove_btn.config(
            state="normal" if len(self.moving_imgs) > 1 else "disabled")
        self._add_btn.config(state="normal")
        print(f"[INFO] Moving 슬롯 정리: {len(self.moving_imgs)}개")

    def _update_mask_label(self) -> None:
        """마스크 상태 라벨을 갱신한다."""
        if self.fixed_mask is None:
            self.mask_label.config(text="  마스크 미선택")
            return
        parts = [f"F:{np.sum(self.fixed_mask > 0) / self.fixed_mask.size * 100:.0f}%"]
        for i, m in enumerate(self.moving_masks):
            if m is not None:
                pct = np.sum(m > 0) / m.size * 100
                parts.append(f"M{i+1}:{pct:.0f}%")
        self.mask_label.config(text="  " + " | ".join(parts))

    def _show_fixed_preview(self) -> None:
        """고정상 미리보기를 결과 영역에 표시한다."""
        self._photo_refs.clear()
        photo = np_to_photo(self.fixed_img)
        self._photo_refs.append(photo)
        self.registered_label.config(image=photo, text="")
        self.falsecolor_label.config(image="", text="(등록 전)")
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", "이미지 로드 → Select Masks (SAM2) → Register")
        self.metrics_text.configure(state="disabled")

    # ── SAM2 마스크 선택 (메인 스레드) ──

    def _select_masks(self) -> None:
        """SAM2를 사용하여 마스크를 선택한다.

        빈 슬롯은 자동으로 정리된다.
        """
        if self.fixed_img is None:
            messagebox.showwarning("Warning", "Fixed 이미지를 먼저 선택하세요.")
            return

        loaded_indices = [i for i, img in enumerate(self.moving_imgs)
                          if img is not None]
        if not loaded_indices:
            messagebox.showwarning("Warning", "Moving 이미지를 최소 1개 선택하세요.")
            return

        # Auto-compact: remove empty slots
        if len(loaded_indices) < len(self.moving_imgs):
            self._compact_moving_slots(loaded_indices)

        print("[INFO] SAM2 predictor 로딩...")
        predictor = load_sam2_predictor()

        # Build image list: [fixed, moving1, moving2, ...]
        images = []
        titles = []
        f_resized, f_scale = resize_for_sam(self.fixed_img, SAM2_MAX_SIDE)
        images.append(f_resized)
        titles.append("Fixed")
        moving_scale_map: list[tuple[int, float]] = []  # (orig_idx, scale)
        for i, img in enumerate(self.moving_imgs):
            if img is not None:
                m_resized, m_scale = resize_for_sam(img, SAM2_MAX_SIDE)
                images.append(m_resized)
                titles.append(f"Moving{i + 1}")
                moving_scale_map.append((i, m_scale))

        print(f"[INFO] {len(images)}개 이미지 마스크 선택...")
        masks, raw_anchors = select_multi_mask_interactive(
            images, titles, predictor)

        if all(m is None for m in masks):
            print("[WARN] 마스크 미완료")
            return

        # Assign fixed mask
        if masks[0] is not None:
            oh, ow = self.fixed_img.shape[:2]
            self.fixed_mask = upscale_mask(masks[0], oh, ow) * 255
        else:
            self.fixed_mask = None

        # Assign moving masks
        mask_idx = 1
        for i in range(len(self.moving_imgs)):
            if self.moving_imgs[i] is not None:
                if mask_idx < len(masks) and masks[mask_idx] is not None:
                    oh, ow = self.moving_imgs[i].shape[:2]
                    self.moving_masks[i] = upscale_mask(
                        masks[mask_idx], oh, ow) * 255
                else:
                    self.moving_masks[i] = None
                mask_idx += 1

        self._best_match_hint = None
        self._update_mask_label()

        # 앵커 좌표를 SAM 리사이즈 → 원본 좌표로 변환
        self.anchor_points_per_moving.clear()
        for img_idx, pairs in raw_anchors.items():
            ms_idx = img_idx - 1  # images[0]=Fixed, 이후=Moving 순
            if ms_idx < 0 or ms_idx >= len(moving_scale_map):
                continue
            orig_mi, m_scale = moving_scale_map[ms_idx]
            converted = []
            for fx, fy, mx, my in pairs:
                converted.append((fx / f_scale, fy / f_scale,
                                  mx / m_scale, my / m_scale))
            self.anchor_points_per_moving[orig_mi] = converted
        self._update_anchor_label()
        print("[INFO] 마스크 선택 완료")

    # ── 리사이즈 가능 이미지 팝업 유틸 ──

    def _resizable_image(self, parent: tk.Widget,
                         pil_img: Image.Image) -> tk.Label:
        """부모 창 크기에 맞게 자동 스케일되는 이미지 라벨을 생성한다.

        Args:
            parent: 부모 위젯.
            pil_img: PIL 이미지.

        Returns:
            자동 스케일링 tk.Label.
        """
        lbl = tk.Label(parent)
        lbl.pack(fill="both", expand=True)
        lbl._orig_pil = pil_img
        lbl._orig_w = pil_img.width
        lbl._orig_h = pil_img.height

        def _on_resize(event):
            w_ = event.width
            h_ = event.height
            if w_ < 10 or h_ < 10:
                return
            s = min(w_ / lbl._orig_w, h_ / lbl._orig_h)
            nw = int(lbl._orig_w * s)
            nh = int(lbl._orig_h * s)
            resized = lbl._orig_pil.resize((nw, nh), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            lbl.config(image=photo)
            lbl._photo = photo

        lbl.bind("<Configure>", _on_resize)
        return lbl

    # ── 2×4 그리드 유틸 ──

    def _grid_dims(self) -> tuple[int, int]:
        """결과 그리드 차원을 반환한다.

        Returns:
            (행 수, 열 수) 튜플.
        """
        n_rows = len(MAX_SIDES) * len(MASK_SIGMAS)
        n_cols = len(CONF_LEVELS) * len(CLAHE_CLIPS)
        return n_rows, n_cols

    def _idx_to_rc(self, idx: int) -> tuple[int, int]:
        """결과 인덱스를 그리드 (row, col) 좌표로 변환한다.

        Args:
            idx: 결과 리스트 인덱스.

        Returns:
            (row, col) 튜플.
        """
        n_rows, _ = self._grid_dims()
        return idx % n_rows, idx // n_rows

    def _screen_figsize(self, dpi: int = 85) -> tuple[float, float]:
        """화면 크기에 맞는 matplotlib figure 크기를 계산한다.

        Args:
            dpi: Figure DPI.

        Returns:
            (width, height) 인치 단위 튜플.
        """
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        fig_w = max(12, (sw - 60) / dpi)
        fig_h = max(5, (sh - 160) / dpi)
        return fig_w, fig_h

    def _add_grid_headers(self, axes: np.ndarray,
                          n_rows: int, n_cols: int) -> None:
        """결과 그리드에 열/행 헤더를 추가한다.

        Args:
            axes: matplotlib axes 2D 배열.
            n_rows: 행 수.
            n_cols: 열 수.
        """
        n_clip = len(CLAHE_CLIPS)
        for col_idx in range(n_cols):
            conf_idx = col_idx // n_clip
            clip_idx = col_idx % n_clip
            hdr = f"c>{CONF_LEVELS[conf_idx]}"
            if n_clip > 1:
                hdr += f"\ncl={CLAHE_CLIPS[clip_idx]}"
            axes[0, col_idx].text(
                0.5, 1.2, hdr,
                transform=axes[0, col_idx].transAxes,
                ha='center', fontsize=8, fontweight='bold')

        n_sig = len(MASK_SIGMAS)
        for row_idx in range(n_rows):
            ms_idx = row_idx // n_sig
            sig_idx = row_idx % n_sig
            lbl = f"ms={MAX_SIDES[ms_idx]}"
            if n_sig > 1:
                lbl += f"\nσ={MASK_SIGMAS[sig_idx]}"
            axes[row_idx, 0].text(
                -0.05, 0.5, lbl,
                transform=axes[row_idx, 0].transAxes,
                ha='right', va='center', fontsize=7, fontweight='bold')

    # ── Anchor Points ──

    def _update_anchor_label(self) -> None:
        """앵커 포인트 상태 라벨을 갱신한다 (UI 숨김)."""
        pass

    # ── Register (별도 스레드 → 2×4 비교 + 메인 결과) ──

    def _run_register(self) -> None:
        """정합을 실행한다.

        유효한 이동상 각각에 대해 별도 스레드에서 register_test를
        순차 실행한다.
        """
        if self.fixed_img is None:
            messagebox.showwarning("Warning", "Fixed 이미지를 먼저 선택하세요.")
            return

        valid = [i for i in range(len(self.moving_imgs))
                 if self.moving_imgs[i] is not None
                 and self.moving_masks[i] is not None]
        if self.fixed_mask is None or not valid:
            messagebox.showwarning("Warning", "마스크를 먼저 선택하세요.")
            return

        self.register_btn.config(state="disabled")
        self._add_btn.config(state="disabled")
        self._remove_btn.config(state="disabled")
        for _, _, btn in self.moving_rows:
            btn.config(state="disabled")
        self.status_label.config(text="Register 진행 중...")
        self._multi_regtest_results.clear()
        self._multi_regtest_selected.clear()

        def worker():
            t0 = time.time()
            all_results = {}
            try:
                for step, mi in enumerate(valid):
                    print(f"[Register] Moving{mi + 1} 시작 "
                          f"({step + 1}/{len(valid)})...")
                    mi_anchors = self.anchor_points_per_moving.get(mi)
                    results = register_test(
                        self.fixed_img, self.moving_imgs[mi],
                        self.fixed_mask, self.moving_masks[mi],
                        anchor_points=mi_anchors or None)
                    all_results[mi] = results
                elapsed = time.time() - t0
                self.root.after(0, self._show_multi_register_results,
                                all_results, valid, elapsed)
            except Exception as e:
                elapsed = time.time() - t0
                print(f"[ERROR] Register failed: {e}")
                self.root.after(0, self._on_register_done, elapsed)

        gc.collect()
        threading.Thread(target=worker, daemon=True).start()

    def _restore_ui_after_register(self) -> None:
        """정합 완료 후 UI 버튼 상태를 복원한다."""
        self.register_btn.config(state="normal")
        self._add_btn.config(state="normal")
        self._remove_btn.config(
            state="normal" if len(self.moving_imgs) > 1 else "disabled")
        for _, _, btn in self.moving_rows:
            btn.config(state="normal")

    def _on_register_done(self, elapsed: float) -> None:
        """정합 오류 시 UI 상태를 복원한다.

        Args:
            elapsed: 경과 시간 (초).
        """
        self._restore_ui_after_register()
        self.status_label.config(text=f"Register 오류 ({elapsed:.1f}s)")

    def _show_multi_register_results(
            self, all_results: dict[int, list[dict]],
            valid_indices: list[int], elapsed: float) -> None:
        """다중 이동상 정합 결과를 탭 형태로 표시한다.

        Args:
            all_results: {이동상 인덱스: 결과 리스트} 딕셔너리.
            valid_indices: 유효한 이동상 인덱스 리스트.
            elapsed: 전체 경과 시간 (초).
        """
        self._restore_ui_after_register()
        self._multi_regtest_results = all_results
        n_rows, n_cols = self._grid_dims()

        # Find best per moving
        rank = {'pass': 2, 'warn': 1, 'fail': 0}
        for mi in valid_indices:
            results = all_results[mi]
            best_idx = 0
            best_score = (-1, -1, -1)
            for idx, r in enumerate(results):
                score = (rank.get(r['status'], 0),
                         r['conf_threshold'],
                         r.get('metrics', {}).get('n_inlier', 0))
                if score > best_score:
                    best_score = score
                    best_idx = idx
            self._multi_regtest_selected[mi] = best_idx

        n_pass_total = sum(
            sum(1 for r in all_results[mi]
                if r['status'] in ('pass', 'warn'))
            for mi in valid_indices)
        n_total = sum(len(all_results[mi]) for mi in valid_indices)

        self.status_label.config(
            text=f"Register: {len(valid_indices)} moving, "
                 f"{n_pass_total}/{n_total} pass ({elapsed:.1f}s)")

        # Show first moving's best in main result
        first_mi = valid_indices[0]
        first_best = all_results[first_mi][
            self._multi_regtest_selected[first_mi]]
        self._show_main_result(first_best)

        # Create Toplevel with Notebook
        top = tk.Toplevel(self.root)
        top.title(f"Register Results ({len(valid_indices)} moving)")

        bar = ttk.Frame(top, padding=3)
        bar.pack(fill="x")
        if len(valid_indices) > 1:
            ttk.Button(bar, text="Save All Selected",
                       command=lambda: self._save_all_results(top)
                       ).pack(side="right", padx=5)

        notebook = ttk.Notebook(top)
        notebook.pack(fill="both", expand=True)

        for mi in valid_indices:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=f"Moving{mi + 1}")
            self._build_register_tab(tab, mi, all_results[mi], top)

        def on_tab_change(event):
            tab_idx = notebook.index(notebook.select())
            mi = valid_indices[tab_idx]
            sel_idx = self._multi_regtest_selected.get(mi, 0)
            self._show_main_result(all_results[mi][sel_idx])

        notebook.bind("<<NotebookTabChanged>>", on_tab_change)
        top.state('zoomed')

    def _format_selection_info(self, r: dict) -> str:
        """선택된 결과의 정보 문자열을 생성한다.

        Args:
            r: 결과 딕셔너리.

        Returns:
            ``"Selected: conf>... ms=... (...)"`` 형식 문자열.
        """
        return (f"Selected: conf>{r['conf_threshold']} "
                f"ms={r['max_side']} ({r['status'].upper()}, "
                f"inlier={r.get('metrics', {}).get('n_inlier', '?')})")

    def _build_register_tab(self, tab_frame: ttk.Frame, moving_idx: int,
                            results: list[dict],
                            parent_window: tk.Toplevel) -> None:
        """결과 탭 하나를 구성한다.

        2xN 그리드 캔버스, 정보 바, Save Selected 버튼을 포함한다.

        Args:
            tab_frame: 탭 프레임.
            moving_idx: 이동상 인덱스.
            results: 해당 이동상의 결과 리스트.
            parent_window: 부모 Toplevel 창.
        """
        n_total = len(results)
        n_rows, n_cols = self._grid_dims()
        best_idx = self._multi_regtest_selected[moving_idx]

        # Info bar inside tab
        tab_bar = ttk.Frame(tab_frame, padding=3)
        tab_bar.pack(fill="x")

        info_var = tk.StringVar(
            value=self._format_selection_info(results[best_idx]))
        ttk.Label(tab_bar, textvariable=info_var).pack(side="left", padx=5)
        ttk.Button(tab_bar, text="Save Selected",
                   command=lambda: self._save_regtest_selected(
                       parent_window, moving_idx)
                   ).pack(side="right", padx=5)

        # Agg render
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        matplotlib.rcParams['axes.unicode_minus'] = False
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        dpi = 85
        fig_w, fig_h = self._screen_figsize(dpi)
        fig = Figure(figsize=(fig_w, fig_h), dpi=dpi)
        axes = fig.subplots(n_rows, n_cols)

        for idx, r in enumerate(results):
            row, col = self._idx_to_rc(idx)
            ax = axes[row, col]
            is_best = (idx == best_idx)

            if r['false_color'] is not None:
                ax.imshow(r['false_color'])
                inlier = r.get('metrics', {}).get('n_inlier', 0)
                title = f"{r['status'].upper()[0]}{inlier}"
            else:
                ax.text(0.5, 0.5, "F", transform=ax.transAxes,
                        ha='center', fontsize=8, color='red')
                ax.set_facecolor('#f0f0f0')
                title = f"F{r['n_matches']}"

            if is_best and r['false_color'] is not None:
                ax.set_title(f"★{title}", fontsize=7, fontweight='bold',
                             color='#FF6600', backgroundcolor='#FFFACD')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#FF6600')
                    spine.set_linewidth(2)
                    spine.set_visible(True)
            else:
                ax.set_title(title, fontsize=7, fontweight='bold')
            ax.axis('off')

        self._add_grid_headers(axes, n_rows, n_cols)

        n_pass = sum(1 for r in results if r['status'] in ('pass', 'warn'))
        fig.suptitle(
            f"Moving{moving_idx + 1}: {n_pass}/{n_total} pass  "
            f"★best — click to select",
            fontsize=11, fontweight='bold', y=0.99)
        fig.tight_layout(rect=[0.03, 0, 1, 0.96])

        canvas_agg = FigureCanvasAgg(fig)
        canvas_agg.draw()

        fig_px_w = int(fig.get_figwidth() * fig.dpi)
        fig_px_h = int(fig.get_figheight() * fig.dpi)
        cell_boxes = []
        for idx in range(n_total):
            row, col = self._idx_to_rc(idx)
            ax = axes[row, col]
            bbox = ax.get_position()
            x0 = int(bbox.x0 * fig_px_w)
            y0 = int((1 - bbox.y1) * fig_px_h)
            x1 = int(bbox.x1 * fig_px_w)
            y1 = int((1 - bbox.y0) * fig_px_h)
            cell_boxes.append((x0, y0, x1, y1))

        buf = canvas_agg.buffer_rgba()
        img_array = np.asarray(buf)[:, :, :3].copy()

        pil_img = Image.fromarray(img_array)
        canvas = tk.Canvas(tab_frame)
        canvas.pack(fill="both", expand=True)
        canvas._orig_pil = pil_img
        canvas._orig_w = pil_img.width
        canvas._orig_h = pil_img.height
        canvas._cell_boxes = cell_boxes
        canvas._best_idx = best_idx
        canvas._sel_idx = best_idx
        canvas._scale = 1.0

        def _redraw_canvas(event=None):
            cw_ = canvas.winfo_width()
            ch_ = canvas.winfo_height()
            if cw_ < 10 or ch_ < 10:
                return
            s = min(cw_ / canvas._orig_w, ch_ / canvas._orig_h)
            canvas._scale = s
            nw_ = int(canvas._orig_w * s)
            nh_ = int(canvas._orig_h * s)
            resized = canvas._orig_pil.resize(
                (nw_, nh_), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas._photo = photo
            bx0, by0, bx1, by1 = canvas._cell_boxes[
                canvas._best_idx]
            canvas.create_rectangle(
                bx0*s, by0*s, bx1*s, by1*s,
                outline="#FF6600", width=5, tags="best_rect")
            si = canvas._sel_idx
            if si != canvas._best_idx:
                sx0, sy0, sx1, sy1 = canvas._cell_boxes[si]
                canvas.create_rectangle(
                    sx0*s, sy0*s, sx1*s, sy1*s,
                    outline="#0066CC", width=6, tags="sel_rect")
            canvas.tag_raise("sel_rect")

        canvas.bind("<Configure>", _redraw_canvas)

        def on_click(event):
            s = canvas._scale
            ox, oy = event.x / s, event.y / s
            for idx, (x0, y0, x1, y1) in enumerate(cell_boxes):
                if x0 <= ox <= x1 and y0 <= oy <= y1:
                    self._multi_regtest_selected[moving_idx] = idx
                    canvas._sel_idx = idx
                    r = results[idx]
                    info_var.set(self._format_selection_info(r))
                    canvas.delete("sel_rect")
                    canvas.create_rectangle(
                        x0*s, y0*s, x1*s, y1*s,
                        outline="#0066CC", width=6, tags="sel_rect")
                    canvas.tag_raise("sel_rect")
                    self._show_main_result(r)
                    break

        canvas.bind("<Button-1>", on_click)

    # ── 메인 결과 영역 표시 ──

    def _show_main_result(self, entry: dict) -> None:
        """결과를 메인 결과 영역에 표시한다.

        Args:
            entry: 결과 딕셔너리.
        """
        self.result = entry
        self._photo_refs.clear()

        reg_img = entry.get('registered_img')
        fc_img = entry.get('false_color')
        status = entry.get('status', 'fail')
        metrics = entry.get('metrics', {})

        if reg_img is not None:
            photo_reg = np_to_photo(reg_img)
            self._photo_refs.append(photo_reg)
            self.registered_label.config(image=photo_reg, text="")

            if fc_img is not None:
                photo_fc = np_to_photo(fc_img)
                self._photo_refs.append(photo_fc)
                self.falsecolor_label.config(image=photo_fc, text="")
        else:
            self.registered_label.config(image="", text="정합 실패")
            self.falsecolor_label.config(image="", text="")

        # 메트릭
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        lines = [f"status: {status}",
                 f"conf>{entry.get('conf_threshold', '?')} "
                 f"ms={entry.get('max_side', '?')}"]
        for k in ['n_inlier', 'inlier_ratio', 'reproj_median',
                   'rotation_deg', 'scale']:
            v = metrics.get(k)
            if v is not None:
                if isinstance(v, float):
                    lines.append(f"{k}: {v:.4f}")
                else:
                    lines.append(f"{k}: {v}")
        self.metrics_text.insert("1.0", "  |  ".join(lines))
        self.metrics_text.configure(state="disabled")

    # ── 선택 결과 저장 ──

    def _save_regtest_selected(self, parent_window: tk.Toplevel,
                               moving_idx: int) -> None:
        """선택된 결과를 파일로 저장한다.

        Args:
            parent_window: 부모 Toplevel 창.
            moving_idx: 이동상 인덱스.
        """
        idx = self._multi_regtest_selected.get(moving_idx)
        results = self._multi_regtest_results.get(moving_idx)
        if idx is None or results is None:
            return
        r = results[idx]
        reg_img = r.get('registered_img')
        if reg_img is None:
            messagebox.showwarning("Warning",
                                   "선택한 조합에 정합 결과가 없습니다.",
                                   parent=parent_window)
            return

        f_name = os.path.splitext(
            os.path.basename(self.fixed_entry.get()))[0]
        m_name = os.path.splitext(
            os.path.basename(self.moving_entries[moving_idx].get()))[0]
        default_name = f"{f_name}_R_{m_name}.jpg"

        path = filedialog.asksaveasfilename(
            parent=parent_window,
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if not path:
            return
        img_bgr = cv2.cvtColor(reg_img, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(path, img_bgr)
        if ok:
            print(f"[INFO] Moving{moving_idx + 1} 결과 저장: {path}")
        else:
            print(f"[ERROR] 저장 실패: {path}")
        self._show_main_result(r)

    def _save_all_results(self, parent_window: tk.Toplevel) -> None:
        """모든 이동상의 선택된 결과를 폴더에 일괄 저장한다.

        Args:
            parent_window: 부모 Toplevel 창.
        """
        folder = filedialog.askdirectory(
            parent=parent_window,
            title="저장할 폴더 선택")
        if not folder:
            return

        f_name = os.path.splitext(
            os.path.basename(self.fixed_entry.get()))[0]
        saved = 0
        used_names: set[str] = set()

        for mi, results in self._multi_regtest_results.items():
            sel_idx = self._multi_regtest_selected.get(mi, 0)
            r = results[sel_idx]
            reg_img = r.get('registered_img')
            if reg_img is None:
                print(f"[WARN] Moving{mi + 1}: 정합 결과 없음, 건너뜀")
                continue

            m_name = os.path.splitext(
                os.path.basename(self.moving_entries[mi].get()))[0]
            filename = f"{f_name}_R_{m_name}.jpg"
            # 파일명 충돌 방지
            if filename in used_names:
                base = f"{f_name}_R_{m_name}"
                n = 2
                while f"{base}_{n}.jpg" in used_names:
                    n += 1
                filename = f"{base}_{n}.jpg"
            used_names.add(filename)
            path = os.path.join(folder, filename)

            img_bgr = cv2.cvtColor(reg_img, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(path, img_bgr)
            if ok:
                print(f"[INFO] 저장: {path}")
                saved += 1
            else:
                print(f"[ERROR] 저장 실패: {path}")

        print(f"[INFO] 전체 저장 완료: {saved}개 파일")


def main() -> None:
    """GUI를 실행한다."""
    root = tk.Tk()
    MainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
