"""[임시] 치아를 한 번 선택해, 치아 영역과 그 반전 영역으로 각각 색조를 절반씩 맞춘다.

- 마스크 선택은 기존 SAM2 UI(select_dual_mask_interactive) 그대로.
    좌클릭 = 전경점 / 우클릭 = 배경점 / z = 개체 확정(여러 번 누적) / x = 초기화
    c = 완료 / q = 취소
- 선택 1회로 두 영역을 만든다:  치아 = 선택 마스크,  치은 = 그 반전(~마스크)
- 각 영역마다 두 이미지의 Lab 평균·표준편차 중간지점을 목표로, 양쪽이 절반씩 이동.
  보정 계수는 해당 영역에서 뽑고 이미지 전체에 적용한다.
- 마스크는 .npy 로 저장한다(재사용·반전·조합 가능).

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
FONT = r"C:\Windows\Fonts\malgunbd.ttf"

os.makedirs(MASK_DIR, exist_ok=True)

load = lambda p: np.array(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))  # noqa: E731
to_lab = lambda u8: cv2.cvtColor(u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)  # noqa: E731
to_rgb = lambda l: np.round(np.clip(cv2.cvtColor(l, cv2.COLOR_LAB2RGB), 0, 1) * 255).astype(np.uint8)  # noqa: E731,E741
dE = lambda x, y: float(np.sqrt(((x - y) ** 2).sum()))  # noqa: E731


def stats(lab, mask):
    px = lab[mask.astype(bool)]
    return px.mean(0), px.std(0)


a_rgb, b_rgb = load(A_PATH), load(B_PATH)
print(f"A {a_rgb.shape}  B {b_rgb.shape}")
print("두 이미지에서 **치아**를 선택 → z 로 확정(여러 번) → c 로 완료\n")

sam = load_sam2_predictor()
mask_a, mask_b = select_dual_mask_interactive(a_rgb, b_rgb, sam)
if mask_a is None or mask_b is None:
    print("두 이미지 모두에서 선택해야 한다. 취소됨.")
    raise SystemExit(1)

np.save(os.path.join(MASK_DIR, "mask_A_치아.npy"), mask_a.astype(np.uint8))
np.save(os.path.join(MASK_DIR, "mask_B_치아.npy"), mask_b.astype(np.uint8))
print(f"치아 화소 — A {int(mask_a.sum()):,} / B {int(mask_b.sum()):,}")
print(f"마스크 저장: {MASK_DIR}\n")

lab_a, lab_b = to_lab(a_rgb), to_lab(b_rgb)
font = ImageFont.truetype(FONT, 120)

for region, ma, mb in (("치아", mask_a, mask_b),
                       ("치은", 1 - mask_a, 1 - mask_b)):     # 치은 = 치아 반전
    print(f"───── 영역: {region}  (A {int(ma.sum()):,} / B {int(mb.sum()):,} 화소)")
    mA, sA = stats(lab_a, ma)
    mB, sB = stats(lab_b, mb)
    tgt_m, tgt_s = (mA + mB) / 2.0, (sA + sB) / 2.0
    print(f"{'':7}{'L*':>9}{'a*':>9}{'b*':>9}")
    print(f"A 평균 {mA[0]:9.2f}{mA[1]:9.2f}{mA[2]:9.2f}   sd {sA.round(2)}")
    print(f"B 평균 {mB[0]:9.2f}{mB[1]:9.2f}{mB[2]:9.2f}   sd {sB.round(2)}")
    print(f"목표   {tgt_m[0]:9.2f}{tgt_m[1]:9.2f}{tgt_m[2]:9.2f}")
    print(f"보정 전 ΔE = {dE(mA, mB):.2f}")

    adj, after = {}, {}
    for tag, lab, m, s, mask, src in (("A", lab_a, mA, sA, ma, A_PATH),
                                      ("B", lab_b, mB, sB, mb, B_PATH)):
        t_m, t_s = tgt_m.copy(), tgt_s.copy()
        if not MATCH_L:
            t_m[0], t_s[0] = m[0], s[0]
        gain = np.where(s > 1e-6, t_s / np.maximum(s, 1e-6), 1.0)
        rgb = to_rgb((lab - m) * gain + t_m)
        adj[tag] = rgb
        after[tag] = stats(to_lab(rgb), mask)[0]
        out = os.path.join(
            OUT_DIR,
            f"{os.path.splitext(os.path.basename(src))[0]}_matched_{region}.jpg")
        Image.fromarray(rgb).save(out, quality=95, subsampling=0)
        print(f"  {tag} 저장 {os.path.basename(out)}   gain={gain.round(3)}")
    print(f"보정 후 ΔE = {dE(after['A'], after['B']):.2f}\n")

    for label, lt, rt, ltxt, rtxt in (
            ("니콘프로파일-vs-캐논프로파일", "A", "B", A_NAME, B_NAME),
            ("캐논프로파일-vs-니콘프로파일", "B", "A", B_NAME, A_NAME)):
        L, R = Image.fromarray(adj[lt]), Image.fromarray(adj[rt])
        w, h = L.size
        comp = L.copy()
        comp.paste(R.crop((SPLIT_X, 0, w, h)), (SPLIT_X, 0))
        d = ImageDraw.Draw(comp)
        for txt, cx in ((ltxt, SPLIT_X // 2), (rtxt, SPLIT_X + (w - SPLIT_X) // 2)):
            d.text((cx, h - 70), txt, font=font, fill=(255, 255, 255),
                   anchor="md", stroke_width=8, stroke_fill=(0, 0, 0))
        comp.save(os.path.join(
            OUT_DIR, f"비교_1_정면_{label}_정중선_matched_{region}.jpg"),
            quality=95, subsampling=0)
    print(f"  합성 2장 저장 ({region})\n")
