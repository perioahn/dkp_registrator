"""[임시] 부위별 이원 색조 보정 — 치아와 치은에 서로 다른 보정을 적용한다.

선택 순서 (모두 두 이미지에 대해):
  ① 6전치        SAM2 클릭   → 치아 보정량을 산출할 표본
  ② 모든 치아    SAM2 클릭   → ①에서 구한 보정을 적용할 영역
  ③ 치은 박스    사각형 드래그 → (박스 ∩ ②의 여집합) = 치은 보정량 산출 표본

적용:
  ② 영역        ← ① 기준 보정
  ② 의 여집합   ← ③ 기준 보정
  경계는 FEATHER_SIGMA 만큼 부드럽게 섞는다(0이면 칼경계).

각 보정은 두 이미지의 Lab 평균·표준편차 **중간지점**을 목표로 양쪽이 절반씩 이동한다.
마스크는 전부 .npy 로 저장한다.

이번 작업용 일회성 스크립트. 엔진 코드는 건드리지 않는다.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam2_mask import load_sam2_predictor, select_dual_mask_interactive  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
D = r"C:\Users\User\Dropbox\00000_사진\프로파일\moim"
OUT_DIR = os.path.join(D, "비교", "새 폴더")
MASK_DIR = os.path.join(OUT_DIR, "masks")
A_PATH = os.path.join(D, "1_정면_nikon_profile_applied.jpg")
B_PATH = os.path.join(D, "1_정면_canon_profile_applied.jpg")
A_NAME, B_NAME = "니콘 프로파일", "캐논 프로파일"
SPLIT_X = 2409
MATCH_L = True
FEATHER_SIGMA = 8.0      # 치아/치은 경계 블렌딩. 0 이면 칼경계
FONT = r"C:\Windows\Fonts\malgunbd.ttf"
TAG = "부위별"

os.makedirs(MASK_DIR, exist_ok=True)

load = lambda p: np.array(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))  # noqa: E731
to_lab = lambda u8: cv2.cvtColor(u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)  # noqa: E731
to_rgb = lambda x: np.round(np.clip(cv2.cvtColor(x, cv2.COLOR_LAB2RGB), 0, 1) * 255).astype(np.uint8)  # noqa: E731
dE = lambda x, y: float(np.sqrt(((x - y) ** 2).sum()))  # noqa: E731


def stats(lab, mask):
    px = lab[mask.astype(bool)]
    if px.size == 0:
        raise SystemExit("선택 영역이 비어 있다.")
    return px.mean(0), px.std(0)


def half_target(sa, sb):
    """(평균, 표준편차) 쌍 두 개의 중간지점."""
    (mA, sA), (mB, sB) = sa, sb
    return (mA + mB) / 2.0, (sA + sB) / 2.0


def corrected(lab, m, s, tgt_m, tgt_s):
    t_m, t_s = tgt_m.copy(), tgt_s.copy()
    if not MATCH_L:
        t_m[0], t_s[0] = m[0], s[0]
    gain = np.where(s > 1e-6, t_s / np.maximum(s, 1e-6), 1.0)
    return (lab - m) * gain + t_m, gain


def select_box(img, title):
    """사각형 박스 하나를 드래그로 받는다. c 또는 Enter 로 확정."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(f"{title} — 드래그로 박스 지정 후 c / Enter")
    box = {}

    def onselect(ec, er):
        box["x0"], box["x1"] = sorted((int(ec.xdata), int(er.xdata)))
        box["y0"], box["y1"] = sorted((int(ec.ydata), int(er.ydata)))
        ax.set_title(f"{title} — ({box['x0']},{box['y0']})-({box['x1']},{box['y1']})"
                     "   c / Enter 로 확정")
        fig.canvas.draw_idle()

    rs = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda e: plt.close(fig) if e.key in ("c", "enter", "ㅊ") else None)
    plt.show()
    del rs
    if not box:
        raise SystemExit("박스가 지정되지 않았다.")
    return box


a_rgb, b_rgb = load(A_PATH), load(B_PATH)
H, W = a_rgb.shape[:2]
print(f"A {a_rgb.shape}  B {b_rgb.shape}\n")
sam = load_sam2_predictor()

print("① 6전치를 선택하시오 (양쪽 이미지)  — 클릭 → z → c")
m1a, m1b = select_dual_mask_interactive(a_rgb, b_rgb, sam)
if m1a is None or m1b is None:
    raise SystemExit("① 취소됨")
print(f"   ① 6전치  A {int(m1a.sum()):,} / B {int(m1b.sum()):,}\n")

print("② 모든 치아를 선택하시오 (양쪽 이미지)  — 클릭 → z(반복) → c")
m2a, m2b = select_dual_mask_interactive(a_rgb, b_rgb, sam)
if m2a is None or m2b is None:
    raise SystemExit("② 취소됨")
print(f"   ② 모든치아  A {int(m2a.sum()):,} / B {int(m2b.sum()):,}\n")

print("③ 상악 전치부 치은 영역을 박스로 지정하시오 (A → B 순서로 두 번)")
boxes = [select_box(a_rgb, f"③ 치은 박스 — {A_NAME}"),
         select_box(b_rgb, f"③ 치은 박스 — {B_NAME}")]
m3 = []
for box, m2 in zip(boxes, (m2a, m2b)):
    r = np.zeros((H, W), np.uint8)
    r[box["y0"]:box["y1"], box["x0"]:box["x1"]] = 1
    m3.append((r & (1 - m2)).astype(np.uint8))      # 박스에서 치아 제외
m3a, m3b = m3
print(f"   ③ 치은표본  A {int(m3a.sum()):,} / B {int(m3b.sum()):,}\n")

for nm, arr in (("1_6전치_A", m1a), ("1_6전치_B", m1b),
                ("2_모든치아_A", m2a), ("2_모든치아_B", m2b),
                ("3_치은표본_A", m3a), ("3_치은표본_B", m3b)):
    np.save(os.path.join(MASK_DIR, f"mask_{nm}.npy"), arr.astype(np.uint8))
print(f"마스크 6종 저장: {MASK_DIR}\n")

lab_a, lab_b = to_lab(a_rgb), to_lab(b_rgb)
out = {}
for region, ma, mb in (("치아(①기준)", m1a, m1b), ("치은(③기준)", m3a, m3b)):
    sa, sb = stats(lab_a, ma), stats(lab_b, mb)
    tm, ts = half_target(sa, sb)
    print(f"───── {region}")
    print(f"{'':7}{'L*':>9}{'a*':>9}{'b*':>9}")
    print(f"A 평균 {sa[0][0]:9.2f}{sa[0][1]:9.2f}{sa[0][2]:9.2f}")
    print(f"B 평균 {sb[0][0]:9.2f}{sb[0][1]:9.2f}{sb[0][2]:9.2f}")
    print(f"목표   {tm[0]:9.2f}{tm[1]:9.2f}{tm[2]:9.2f}")
    print(f"보정 전 ΔE = {dE(sa[0], sb[0]):.2f}")
    ca, ga = corrected(lab_a, sa[0], sa[1], tm, ts)
    cb, gb = corrected(lab_b, sb[0], sb[1], tm, ts)
    print(f"  gain A={ga.round(3)}  B={gb.round(3)}")
    print(f"보정 후 ΔE = "
          f"{dE(stats(to_lab(to_rgb(ca)), ma)[0], stats(to_lab(to_rgb(cb)), mb)[0]):.2f}\n")
    out[region] = (ca, cb)

# ── 치아=①보정, 그 외=③보정 으로 합성 (②로 가르고 경계는 페더) ──
tooth_a, tooth_b = out["치아(①기준)"]
ging_a, ging_b = out["치은(③기준)"]
final = {}
for tag, m2, t_lab, g_lab, src in (("A", m2a, tooth_a, ging_a, A_PATH),
                                   ("B", m2b, tooth_b, ging_b, B_PATH)):
    alpha = m2.astype(np.float32)
    if FEATHER_SIGMA > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), FEATHER_SIGMA)
    a3 = alpha[..., None]
    rgb = to_rgb(t_lab * a3 + g_lab * (1.0 - a3))
    final[tag] = rgb
    p = os.path.join(
        OUT_DIR, f"{os.path.splitext(os.path.basename(src))[0]}_matched_{TAG}.jpg")
    Image.fromarray(rgb).save(p, quality=95, subsampling=0)
    print(f"{tag} 최종 저장: {os.path.basename(p)}")

font = ImageFont.truetype(FONT, 120)
for label, lt, rt, ltxt, rtxt in (
        ("니콘프로파일-vs-캐논프로파일", "A", "B", A_NAME, B_NAME),
        ("캐논프로파일-vs-니콘프로파일", "B", "A", B_NAME, A_NAME)):
    L, R = Image.fromarray(final[lt]), Image.fromarray(final[rt])
    w, h = L.size
    comp = L.copy()
    comp.paste(R.crop((SPLIT_X, 0, w, h)), (SPLIT_X, 0))
    d = ImageDraw.Draw(comp)
    for txt, cx in ((ltxt, SPLIT_X // 2), (rtxt, SPLIT_X + (w - SPLIT_X) // 2)):
        d.text((cx, h - 70), txt, font=font, fill=(255, 255, 255),
               anchor="md", stroke_width=8, stroke_fill=(0, 0, 0))
    comp.save(os.path.join(
        OUT_DIR, f"비교_1_정면_{label}_정중선_matched_{TAG}.jpg"),
        quality=95, subsampling=0)
print("합성 2장 저장 완료")
