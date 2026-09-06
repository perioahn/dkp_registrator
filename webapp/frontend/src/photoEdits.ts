// 비파괴 조정값 + 저장 시 1회 렌더 파이프라인.
// 순서: EXIF 방향 정규화(로드 시) → rot90/flip/미세회전(흰 패딩) → 크롭 → 밝기/대비 → JPEG q0.95 + EXIF 복사


export interface Edits {
  rot90: number // 0~3 (CW 횟수)
  flipH: boolean
  flipV: boolean
  fineDeg: number // -15 ~ +15
  brightness: number // -100 ~ +100
  contrast: number // -100 ~ +100
  crop: { x: number; y: number; w: number; h: number } | null // fine 회전 후 좌표계, 원본 픽셀 단위
}

export const freshEdits = (): Edits => ({
  rot90: 0, flipH: false, flipV: false, fineDeg: 0,
  brightness: 0, contrast: 0, crop: null,
})

export const isDirty = (e: Edits): boolean =>
  e.rot90 !== 0 || e.flipH || e.flipV || e.fineDeg !== 0 ||
  e.brightness !== 0 || e.contrast !== 0 || e.crop !== null

// ── 라이트룸식 톤 조정 (LUT) ─────────────────────────
// 노출: sRGB→선형 → 2^EV 게인 → 하이라이트 soft shoulder → sRGB 복귀
// 대비: 감마 공간 중간톤 중심 S커브(smoothstep 블렌드)
// CSS brightness()의 단순 곱과 달리 하이라이트가 날아가지 않음.

const srgbToLin = (v: number) =>
  v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)
const linToSrgb = (v: number) =>
  v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055

export function buildLUT(brightness: number, contrast: number): Uint8ClampedArray {
  const ev = (brightness / 100) * 2.5          // 슬라이더 ±100 → ±2.5 스톱
  const gain = Math.pow(2, ev)
  const c = contrast / 100                     // -1 ~ +1
  const KNEE = 0.8                             // shoulder 시작점 (선형 공간)
  const lut = new Uint8ClampedArray(256)
  for (let i = 0; i < 256; i++) {
    let lin = srgbToLin(i / 255) * gain
    if (lin > KNEE) {
      // soft shoulder: KNEE 위는 점근적으로 1에 수렴 (하드 클립 방지)
      const t = lin - KNEE
      lin = KNEE + (1 - KNEE) * (t / (t + (1 - KNEE)))
    }
    let v = linToSrgb(Math.min(1, Math.max(0, lin)))
    if (c > 0) {
      const s = v * v * (3 - 2 * v)            // smoothstep S커브
      v = v * (1 - c) + s * c
    } else if (c < 0) {
      v = v * (1 + c) - 0.5 * c                // 중간회색으로 평탄화
    }
    lut[i] = Math.round(Math.min(1, Math.max(0, v)) * 255)
  }
  return lut
}

/** 캔버스에 LUT 적용 (제자리) */
export function applyLUT(canvas: HTMLCanvasElement, lut: Uint8ClampedArray): void {
  const g = canvas.getContext('2d')!
  const img = g.getImageData(0, 0, canvas.width, canvas.height)
  const d = img.data
  for (let i = 0; i < d.length; i += 4) {
    d[i] = lut[d[i]]
    d[i + 1] = lut[d[i + 1]]
    d[i + 2] = lut[d[i + 2]]
  }
  g.putImageData(img, 0, 0)
}

/** rot90/flip/fine 를 합친 CSS transform (미리보기용) */
export const cssTransform = (e: Edits): string =>
  `rotate(${e.rot90 * 90 + e.fineDeg}deg) scaleX(${e.flipH ? -1 : 1}) scaleY(${e.flipV ? -1 : 1})`

/** rot90+fine 적용 후의 캔버스 크기 (crop 좌표계 기준) */
export function orientedSize(w: number, h: number, e: Edits): { w: number; h: number } {
  const swap = e.rot90 % 2 === 1
  let ow = swap ? h : w
  let oh = swap ? w : h
  if (e.fineDeg !== 0) {
    const rad = (e.fineDeg * Math.PI) / 180
    const cos = Math.abs(Math.cos(rad))
    const sin = Math.abs(Math.sin(rad))
    const nw = Math.round(ow * cos + oh * sin)
    const nh = Math.round(ow * sin + oh * cos)
    ow = nw
    oh = nh
  }
  return { w: ow, h: oh }
}

/** EXIF 방향 반영해 디코드 (브라우저가 처리) */
export async function loadBitmap(file: File): Promise<ImageBitmap> {
  return createImageBitmap(file, { imageOrientation: 'from-image' })
}

/** rot90+flip+fine회전 적용 캔버스 (흰 패딩, core_crop rotate_image_with_padding 이식) */
export function renderOriented(src: ImageBitmap | HTMLCanvasElement, e: Edits): HTMLCanvasElement {
  // 1) rot90 + flip
  let w = src.width, h = src.height
  const swap = e.rot90 % 2 === 1
  const c1 = document.createElement('canvas')
  c1.width = swap ? h : w
  c1.height = swap ? w : h
  const g1 = c1.getContext('2d')!
  g1.translate(c1.width / 2, c1.height / 2)
  g1.rotate((e.rot90 * Math.PI) / 2)
  g1.scale(e.flipH ? -1 : 1, e.flipV ? -1 : 1)
  g1.drawImage(src, -w / 2, -h / 2)
  if (e.fineDeg === 0) return c1

  // 2) 미세회전: 새 크기 = 회전 외접 사각형, 흰색 패딩
  const rad = (e.fineDeg * Math.PI) / 180
  const cos = Math.abs(Math.cos(rad)), sin = Math.abs(Math.sin(rad))
  const c2 = document.createElement('canvas')
  c2.width = Math.round(c1.width * cos + c1.height * sin)
  c2.height = Math.round(c1.width * sin + c1.height * cos)
  const g2 = c2.getContext('2d')!
  g2.fillStyle = '#fff'
  g2.fillRect(0, 0, c2.width, c2.height)
  g2.translate(c2.width / 2, c2.height / 2)
  g2.rotate(rad)
  g2.drawImage(c1, -c1.width / 2, -c1.height / 2)
  return c2
}

export type Matrix3 = [[number, number, number], [number, number, number], [number, number, number]]

/** Pixel-center coordinates, matching OpenCV: p_edit = G p_source.
 * Cropping clips integer pixels and never rescales either axis. */
export function editGeometry(width: number, height: number, edits: Edits) {
  const size = orientedSize(width, height, edits)
  const c = edits.crop
  const crop = c ? {
    x: Math.min(size.w - 1, Math.max(0, Math.round(c.x))),
    y: Math.min(size.h - 1, Math.max(0, Math.round(c.y))),
    w: Math.max(1, Math.min(size.w, Math.round(c.w))),
    h: Math.max(1, Math.min(size.h, Math.round(c.h))),
  } : { x: 0, y: 0, w: size.w, h: size.h }
  crop.w = Math.min(crop.w, size.w - crop.x)
  crop.h = Math.min(crop.h, size.h - crop.y)
  const theta = (edits.rot90 * 90 + edits.fineDeg) * Math.PI / 180
  const cos = Math.cos(theta), sin = Math.sin(theta)
  const fx = edits.flipH ? -1 : 1, fy = edits.flipV ? -1 : 1
  const a = cos * fx, b = -sin * fy, d = sin * fx, e = cos * fy
  const x = (width - 1) / 2, y = (height - 1) / 2
  const G: Matrix3 = [
    [a, b, (size.w - 1) / 2 - a * x - b * y - crop.x],
    [d, e, (size.h - 1) / 2 - d * x - e * y - crop.y],
    [0, 0, 1],
  ]
  // Stable exact quarter-turn entries (including negative zero) aid revision comparisons.
  for (const row of G) for (let i = 0; i < row.length; i++) if (Math.abs(row[i]) < 1e-12) row[i] = 0
  return { G, width: crop.w, height: crop.h, crop }
}

export async function renderEditedPNG(file: File, edits: Edits) {
  const bitmap = await loadBitmap(file)
  try {
    const geometry = editGeometry(bitmap.width, bitmap.height, edits)
    const oriented = renderOriented(bitmap, edits)
    const canvas = document.createElement('canvas')
    canvas.width = geometry.width; canvas.height = geometry.height
    const g = canvas.getContext('2d')!
    g.fillStyle = '#fff'; g.fillRect(0, 0, canvas.width, canvas.height)
    g.drawImage(oriented, -geometry.crop.x, -geometry.crop.y)
    if (edits.brightness || edits.contrast) applyLUT(canvas, buildLUT(edits.brightness, edits.contrast))
    const blob = await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob(value => value ? resolve(value) : reject(new Error('PNG 생성 실패')), 'image/png'))
    return { blob, G: geometry.G, width: canvas.width, height: canvas.height,
      edits: { ...edits, crop: edits.crop ? geometry.crop : null } }
  } finally { bitmap.close() }
}

