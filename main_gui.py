# -*- coding: utf-8 -*-
"""
main_gui.py — 치아 정합 파이프라인 GUI (tkinter)

SAM2 마스크 선택 → Register (8 combo 비교) → 결과 선택/저장
"""

import sys
import os
import io
import gc
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from register import (match_check, register_test, false_color,
                      CONF_LEVELS, MAX_SIDES, CLAHE_CLIPS, MASK_SIGMAS)
from sam2_mask import (load_sam2_predictor, select_mask_interactive,
                       select_dual_mask_interactive,
                       resize_for_sam, upscale_mask, clean_mask)

SAM2_MAX_SIDE = 1024
THUMB_MAX = 400

# ── 유틸리티 ────────────────────────────────────────

def load_image_rgb(path):
    """이미지 파일을 RGB numpy 배열로 로드."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"이미지 로드 실패: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def np_to_photo(img_rgb, max_side=THUMB_MAX):
    """numpy RGB → 썸네일 PhotoImage."""
    h, w = img_rgb.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h))
    return ImageTk.PhotoImage(Image.fromarray(img_rgb))


class TextRedirector(io.TextIOBase):
    """stdout/stderr → Text 위젯 리디렉터."""
    def __init__(self, text_widget):
        self.text = text_widget

    def write(self, s):
        if s:
            try:
                self.text.after(0, self._append, s)
            except RuntimeError:
                pass  # widget destroyed
        return len(s) if s else 0

    def _append(self, s):
        self.text.configure(state="normal")
        self.text.insert("end", s)
        self.text.see("end")
        self.text.configure(state="disabled")

    def flush(self):
        pass


# ── 메인 GUI ────────────────────────────────────────

class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("치아 정합 파이프라인")
        self.root.geometry("900x800")

        # 상태
        self.fixed_img = None    # 원본 해상도 RGB
        self.moving_img = None
        self.fixed_mask = None   # 원본 해상도 uint8 (0/255)
        self.moving_mask = None
        self.result = None       # 현재 표시 중인 결과 entry
        self._photo_refs = []    # PhotoImage 참조 유지
        self._best_match_hint = None
        self._regtest_results = None
        self._regtest_selected_idx = None

        self._build_ui()
        self._redirect_console()

    def _build_ui(self):
        # ── 파일 선택 ──
        file_frame = ttk.LabelFrame(self.root, text="이미지 선택", padding=5)
        file_frame.pack(fill="x", padx=5, pady=3)

        ttk.Label(file_frame, text="Fixed:").grid(row=0, column=0, sticky="w")
        self.fixed_entry = ttk.Entry(file_frame, width=60)
        self.fixed_entry.grid(row=0, column=1, padx=3)
        ttk.Button(file_frame, text="Browse",
                   command=lambda: self._browse("fixed")).grid(row=0, column=2)

        ttk.Label(file_frame, text="Moving:").grid(row=1, column=0, sticky="w")
        self.moving_entry = ttk.Entry(file_frame, width=60)
        self.moving_entry.grid(row=1, column=1, padx=3)
        ttk.Button(file_frame, text="Browse",
                   command=lambda: self._browse("moving")).grid(row=1, column=2)

        # ── SAM2 마스크 ──
        mask_frame = ttk.LabelFrame(self.root, text="SAM2 마스크", padding=5)
        mask_frame.pack(fill="x", padx=5, pady=3)

        ttk.Button(mask_frame, text="Select Masks (SAM2)",
                   command=self._select_masks).pack(side="left")
        self.mask_label = ttk.Label(mask_frame, text="  마스크 미선택")
        self.mask_label.pack(side="left", padx=10)

        # ── 실행/저장 버튼 ──
        btn_frame = ttk.Frame(self.root, padding=5)
        btn_frame.pack(fill="x", padx=5)

        self.register_btn = ttk.Button(btn_frame, text="Register",
                                       command=self._run_register)
        self.register_btn.pack(side="left")

        self.matchcheck_btn = ttk.Button(btn_frame, text="Match Check",
                                         command=self._run_match_check)
        self.matchcheck_btn.pack(side="left", padx=5)

        self.save_btn = ttk.Button(btn_frame, text="Save Result",
                                   command=self._save_result, state="disabled")
        self.save_btn.pack(side="left", padx=10)

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

    def _redirect_console(self):
        redir = TextRedirector(self.console)
        sys.stdout = redir
        sys.stderr = redir

    # ── 파일 브라우즈 ──

    def _browse(self, which):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not path:
            return
        entry = self.fixed_entry if which == "fixed" else self.moving_entry
        entry.delete(0, "end")
        entry.insert(0, path)
        try:
            img = load_image_rgb(path)
            if which == "fixed":
                self.fixed_img = img
                self.fixed_mask = None
            else:
                self.moving_img = img
                self.moving_mask = None
            self._best_match_hint = None
            if self.fixed_mask is None or self.moving_mask is None:
                self.mask_label.config(text="  마스크 미선택")
            print(f"[INFO] {which} 로드: {os.path.basename(path)} "
                  f"({img.shape[1]}x{img.shape[0]})")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ── SAM2 마스크 선택 (메인 스레드) ──

    def _select_masks(self):
        if self.fixed_img is None or self.moving_img is None:
            messagebox.showwarning("Warning", "Fixed/Moving 이미지를 먼저 선택하세요.")
            return

        print("[INFO] SAM2 predictor 로딩...")
        predictor = load_sam2_predictor()

        f_resized, _ = resize_for_sam(self.fixed_img, SAM2_MAX_SIDE)
        m_resized, _ = resize_for_sam(self.moving_img, SAM2_MAX_SIDE)

        print("[INFO] Fixed/Moving 마스크 동시 선택...")
        f_mask, m_mask = select_dual_mask_interactive(
            f_resized, m_resized, predictor)

        if f_mask is None or m_mask is None:
            print("[WARN] 마스크 미완료")
            return

        oh, ow = self.fixed_img.shape[:2]
        self.fixed_mask = upscale_mask(f_mask, oh, ow) * 255

        oh, ow = self.moving_img.shape[:2]
        self.moving_mask = upscale_mask(m_mask, oh, ow) * 255
        self._best_match_hint = None

        f_pct = np.sum(self.fixed_mask > 0) / self.fixed_mask.size * 100
        m_pct = np.sum(self.moving_mask > 0) / self.moving_mask.size * 100
        self.mask_label.config(
            text=f"  Fixed: {f_pct:.1f}%  |  Moving: {m_pct:.1f}%")
        print(f"[INFO] 마스크 선택 완료 — Fixed: {f_pct:.1f}%, Moving: {m_pct:.1f}%")

    # ── 2×4 그리드 유틸 ──

    def _grid_dims(self):
        """2×4 그리드: rows=ms*sigma, cols=conf*clahe."""
        n_rows = len(MAX_SIDES) * len(MASK_SIGMAS)
        n_cols = len(CONF_LEVELS) * len(CLAHE_CLIPS)
        return n_rows, n_cols

    def _idx_to_rc(self, idx):
        """결과 인덱스 → (row, col) for 2×4 grid."""
        n_rows, _ = self._grid_dims()
        return idx % n_rows, idx // n_rows

    def _screen_figsize(self, dpi=85):
        """화면 크기에 맞는 figsize 계산."""
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        fig_w = max(12, (sw - 60) / dpi)
        fig_h = max(5, (sh - 160) / dpi)
        return fig_w, fig_h

    def _add_grid_headers(self, axes, n_rows, n_cols):
        """2×4 그리드 헤더: col=conf, row=ms."""
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

    # ── Match Check (별도 스레드 → matplotlib 시각화) ──

    def _run_match_check(self):
        if self.fixed_img is None or self.moving_img is None:
            messagebox.showwarning("Warning", "이미지를 먼저 선택하세요.")
            return
        if self.fixed_mask is None or self.moving_mask is None:
            messagebox.showwarning("Warning", "SAM2 마스크를 먼저 선택하세요.")
            return

        self.matchcheck_btn.config(state="disabled")
        self.status_label.config(text="Match Check 진행 중...")

        def worker():
            t0 = time.time()
            try:
                results = match_check(
                    self.fixed_img, self.moving_img,
                    self.fixed_mask, self.moving_mask)
                elapsed = time.time() - t0
                self.root.after(0, self._show_match_results, results, elapsed)
            except Exception as e:
                elapsed = time.time() - t0
                print(f"[ERROR] Match Check failed: {e}")
                self.root.after(0, self._on_match_check_done, elapsed)

        gc.collect()
        threading.Thread(target=worker, daemon=True).start()

    def _on_match_check_done(self, elapsed):
        self.matchcheck_btn.config(state="normal")
        self.status_label.config(text=f"Match Check 오류 ({elapsed:.1f}s)")

    def _show_match_results(self, results, elapsed):
        self.matchcheck_btn.config(state="normal")
        n_total = len(results)
        n_rows, n_cols = self._grid_dims()

        # 최적값 탐색
        best_idx = None
        best_conf = -1
        best_n_at_conf = -1
        for idx, r in enumerate(results):
            n = r['n_matches']
            ct = r['conf_threshold']
            if n >= 12:
                if ct > best_conf or (ct == best_conf and n > best_n_at_conf):
                    best_idx = idx
                    best_conf = ct
                    best_n_at_conf = n
        if best_idx is None:
            best_idx = max(range(n_total),
                          key=lambda i: results[i]['n_matches'])

        best_r = results[best_idx]
        self._best_match_hint = (best_r['conf_threshold'], best_r['max_side'],
                                 best_r['clahe_clip'], best_r['mask_sigma'])

        print(f"[Match Check] → 최적: conf>{best_r['conf_threshold']} "
              f"ms={best_r['max_side']} ({best_r['n_matches']}개)")

        self.status_label.config(
            text=f"Match best={best_r['n_matches']} "
                 f"(c>{best_r['conf_threshold']} ms={best_r['max_side']}) "
                 f"({elapsed:.1f}s)")

        # Agg 렌더링 2×4
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        matplotlib.rcParams['axes.unicode_minus'] = False
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        dpi = 85
        fig_w, fig_h = self._screen_figsize(dpi)
        fig = Figure(figsize=(fig_w, fig_h), dpi=dpi)
        axes = fig.subplots(n_rows, n_cols)

        for idx, r in enumerate(results):
            row, col = self._idx_to_rc(idx)
            ax = axes[row, col]
            is_best = (idx == best_idx)

            n = r['n_matches']

            if n == 0:
                ax.text(0.5, 0.5, "0", transform=ax.transAxes,
                        ha='center', fontsize=8, color='red')
                ax.set_facecolor('#f0f0f0')
            else:
                fi = r['fixed_img']
                mi = r['moving_img']
                k0 = r['kpts0'].copy()
                k1 = r['kpts1'].copy()
                cf = r['conf'].copy()

                max_show = 25
                if len(k0) > max_show:
                    top_idx = np.argsort(-cf)[:max_show]
                    k0, k1, cf = k0[top_idx], k1[top_idx], cf[top_idx]

                h1, w1 = fi.shape[:2]
                h2, w2 = mi.shape[:2]
                gap = 4
                max_h = max(h1, h2)
                canvas_img = np.full((max_h, w1 + gap + w2), 128,
                                     dtype=np.uint8)
                canvas_img[:h1, :w1] = fi
                canvas_img[:h2, w1 + gap:] = mi
                ax.imshow(canvas_img, cmap='gray')

                for (x0, y0), (x1, y1), c in zip(k0, k1, cf):
                    color = (0.2, 0.8 * c, 0.2 + 0.8 * (1 - c))
                    ax.plot([x0, x1 + w1 + gap], [y0, y1],
                            color=color, alpha=0.3, lw=0.4)
                ax.scatter(k0[:, 0], k0[:, 1], c='lime', s=2, zorder=5)
                ax.scatter(k1[:, 0] + w1 + gap, k1[:, 1], c='lime', s=2,
                           zorder=5)

            if is_best:
                ax.set_title(f"★{n}", fontsize=7, fontweight='bold',
                             color='#FF6600', backgroundcolor='#FFFACD')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#FF6600')
                    spine.set_linewidth(2)
                    spine.set_visible(True)
            else:
                ax.set_title(f"{n}", fontsize=7, fontweight='bold')
            ax.axis('off')

        self._add_grid_headers(axes, n_rows, n_cols)

        n_loftr = len(MAX_SIDES) * len(CLAHE_CLIPS) * len(MASK_SIGMAS)
        fig.suptitle(
            f"Match Check {n_rows}×{n_cols}  ({elapsed:.1f}s)  "
            f"[LoFTR {n_loftr}회, 총 {n_total} combos]",
            fontsize=12, fontweight='bold', y=0.99)
        fig.tight_layout(rect=[0.03, 0, 1, 0.96])

        canvas_agg = FigureCanvasAgg(fig)
        canvas_agg.draw()
        buf = canvas_agg.buffer_rgba()
        img_array = np.asarray(buf)[:, :, :3]

        pil_img = Image.fromarray(img_array)
        top = tk.Toplevel(self.root)
        top.title(f"Match Check {n_rows}×{n_cols}")
        photo = ImageTk.PhotoImage(pil_img)
        lbl = tk.Label(top, image=photo)
        lbl.image = photo
        lbl.pack(fill="both", expand=True)
        top.state('zoomed')

    # ── Register (별도 스레드 → 2×4 비교 + 메인 결과) ──

    def _run_register(self):
        if self.fixed_img is None or self.moving_img is None:
            messagebox.showwarning("Warning", "이미지를 먼저 선택하세요.")
            return
        if self.fixed_mask is None or self.moving_mask is None:
            messagebox.showwarning("Warning", "SAM2 마스크를 먼저 선택하세요.")
            return

        self.register_btn.config(state="disabled")
        self.status_label.config(text="Register 진행 중...")

        def worker():
            t0 = time.time()
            try:
                results = register_test(
                    self.fixed_img, self.moving_img,
                    self.fixed_mask, self.moving_mask)
                elapsed = time.time() - t0
                self.root.after(0, self._show_register_results,
                                results, elapsed)
            except Exception as e:
                elapsed = time.time() - t0
                print(f"[ERROR] Register failed: {e}")
                self.root.after(0, self._on_register_done, elapsed)

        gc.collect()
        threading.Thread(target=worker, daemon=True).start()

    def _on_register_done(self, elapsed):
        self.register_btn.config(state="normal")
        self.status_label.config(text=f"Register 오류 ({elapsed:.1f}s)")

    def _show_register_results(self, results, elapsed):
        self.register_btn.config(state="normal")
        n_total = len(results)
        n_rows, n_cols = self._grid_dims()

        # 최적값: pass > warn > fail, 높은 conf, 많은 inlier
        _rank = {'pass': 2, 'warn': 1, 'fail': 0}
        best_idx = 0
        best_score = (-1, -1, -1)
        for idx, r in enumerate(results):
            score = (_rank.get(r['status'], 0),
                     r['conf_threshold'],
                     r.get('metrics', {}).get('n_inlier', 0))
            if score > best_score:
                best_score = score
                best_idx = idx

        best_r = results[best_idx]
        if best_r['false_color'] is not None:
            self._best_match_hint = (best_r['conf_threshold'],
                                     best_r['max_side'],
                                     best_r['clahe_clip'],
                                     best_r['mask_sigma'])

        n_pass = sum(1 for r in results if r['status'] in ('pass', 'warn'))

        print(f"[Register] → best: conf>{best_r['conf_threshold']} "
              f"ms={best_r['max_side']} "
              f"({best_r['status'].upper()}, {n_pass}/{n_total} pass)")

        self.status_label.config(
            text=f"Register: {n_pass}/{n_total} pass ({elapsed:.1f}s)")

        # 결과 저장 (interactive 선택용)
        self._regtest_results = results
        self._regtest_selected_idx = best_idx

        # best를 메인 결과 영역에 표시
        self._show_main_result(best_r)

        # Agg 렌더링 2×4
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        matplotlib.rcParams['axes.unicode_minus'] = False
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

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

        fig.suptitle(
            f"Register {n_rows}×{n_cols}  ({elapsed:.1f}s)  "
            f"[{n_pass}/{n_total} pass]   ★best — click to select",
            fontsize=12, fontweight='bold', y=0.99)
        fig.tight_layout(rect=[0.03, 0, 1, 0.96])

        canvas_agg = FigureCanvasAgg(fig)
        canvas_agg.draw()

        # 셀 바운딩박스 (클릭 매핑용)
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

        # Toplevel: 상단 바 + Canvas
        top = tk.Toplevel(self.root)
        top.title(f"Register {n_rows}×{n_cols}")

        bar = ttk.Frame(top, padding=3)
        bar.pack(fill="x")

        sel_r = results[best_idx]
        info_var = tk.StringVar(
            value=f"Selected: conf>{sel_r['conf_threshold']} "
                  f"ms={sel_r['max_side']} ({sel_r['status'].upper()}, "
                  f"inlier={sel_r.get('metrics', {}).get('n_inlier', '?')})")
        ttk.Label(bar, textvariable=info_var).pack(side="left", padx=5)
        ttk.Button(bar, text="Save Selected",
                   command=lambda: self._save_regtest_selected(top)
                   ).pack(side="right", padx=5)

        w, h = img_array.shape[1], img_array.shape[0]
        canvas = tk.Canvas(top, width=w, height=h)
        canvas.pack(fill="both", expand=True)

        pil_img = Image.fromarray(img_array)
        photo = ImageTk.PhotoImage(pil_img)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas._photo = photo

        # best 초기 하이라이트 (오렌지)
        bx0, by0, bx1, by1 = cell_boxes[best_idx]
        canvas.create_rectangle(bx0, by0, bx1, by1,
                                outline="#FF6600", width=3, tags="best_rect")

        def on_click(event):
            cx, cy = event.x, event.y
            for idx, (x0, y0, x1, y1) in enumerate(cell_boxes):
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    self._regtest_selected_idx = idx
                    r = results[idx]
                    self._best_match_hint = (r['conf_threshold'],
                                             r['max_side'],
                                             r['clahe_clip'],
                                             r['mask_sigma'])
                    info_var.set(
                        f"Selected: conf>{r['conf_threshold']} "
                        f"ms={r['max_side']} ({r['status'].upper()}, "
                        f"inlier={r.get('metrics', {}).get('n_inlier', '?')})")
                    canvas.delete("sel_rect")
                    canvas.create_rectangle(x0, y0, x1, y1,
                                            outline="#0066CC", width=3,
                                            tags="sel_rect")
                    # 메인 결과 영역도 업데이트
                    self._show_main_result(r)
                    break

        canvas.bind("<Button-1>", on_click)
        top.state('zoomed')

    # ── 메인 결과 영역 표시 ──

    def _show_main_result(self, entry):
        """Register 결과 entry를 메인 결과 영역에 표시."""
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

            self.save_btn.config(state="normal")
        else:
            self.registered_label.config(image="", text="정합 실패")
            self.falsecolor_label.config(image="", text="")
            self.save_btn.config(state="disabled")

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

    def _save_regtest_selected(self, parent_window):
        idx = self._regtest_selected_idx
        results = self._regtest_results
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
            os.path.basename(self.moving_entry.get()))[0]
        default_name = f"{f_name}_R_{m_name}.jpg"

        path = filedialog.asksaveasfilename(
            parent=parent_window,
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if not path:
            return
        img_bgr = cv2.cvtColor(reg_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_bgr)
        print(f"[INFO] 정합 결과 저장: {path} "
              f"(conf>{r['conf_threshold']} ms={r['max_side']})")

        # 메인 결과 영역도 업데이트
        self._show_main_result(r)

    def _save_result(self):
        if self.result is None or self.result.get('registered_img') is None:
            return
        f_name = os.path.splitext(
            os.path.basename(self.fixed_entry.get()))[0]
        m_name = os.path.splitext(
            os.path.basename(self.moving_entry.get()))[0]
        default_name = f"{f_name}_R_{m_name}.jpg"

        path = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if not path:
            return
        img_bgr = cv2.cvtColor(self.result['registered_img'],
                                cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_bgr)
        print(f"[INFO] 저장 완료: {path}")


# ── 엔트리포인트 ────────────────────────────────────

def main():
    root = tk.Tk()
    MainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
