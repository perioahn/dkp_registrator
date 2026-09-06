<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { buildLUT, orientedSize, renderOriented, type Edits } from '../photoEdits'
interface Photo { id: string; name: string; file: File }

const props = defineProps<{
  photo: Photo
  edits: Edits
  mode: 'view' | 'crop'
  cropRatio: number | null // 크롭모드 비율 강제 (null=자유)
  straighten: boolean
  showOriginal: boolean
}>()
const emit = defineEmits<{ angle: [deg: number]; fineDeg: [deg: number]; error: [message: string] }>()

const canvasEl = ref<HTMLCanvasElement | null>(null)
const rootEl = ref<HTMLDivElement | null>(null)
const available = ref({ w: 1000, h: 700 })
let observer: ResizeObserver | null = null
const space = ref(false)
let loadRevision = 0, disposed = false

// 뷰 상태 (view 모드 전용)
const zoom = ref(1)
const panX = ref(0)
const panY = ref(0)

// 크롭 모드 상태 (미리보기 캔버스 좌표)
const rect = ref({ x: 0, y: 0, w: 100, h: 100 })
const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'] as const

const line = ref<{ x1: number; y1: number; x2: number; y2: number } | null>(null)

// 크롭 모드 커서: 위치별 직관적 표시 (핸들=리사이즈, 안=이동, 밖=회전)
const ROTATE_CURSOR = `url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='22' height='22' viewBox='0 0 24 24'><path d='M12 4a8 8 0 1 1-7.5 5.3' fill='none' stroke='black' stroke-width='4.5' stroke-linecap='round'/><path d='M12 4a8 8 0 1 1-7.5 5.3' fill='none' stroke='white' stroke-width='2.5' stroke-linecap='round'/><path d='M2.5 3.5 L5 10 L11 7 Z' fill='white' stroke='black' stroke-width='1'/></svg>") 11 11, grab`
const HANDLE_CURSOR: Record<string, string> = {
  nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize',
  n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize',
}
const cursor = ref('')

function updateCursor(p: { x: number; y: number }) {
  if (props.straighten) { cursor.value = 'crosshair'; return }
  if (props.mode !== 'crop') { cursor.value = ''; return }
  const h = hitHandle(p)
  cursor.value = h ? HANDLE_CURSOR[h] : inRect(p) ? 'move' : ROTATE_CURSOR
}

let bmp: ImageBitmap | null = null
let naturalW = 0
let naturalH = 0
const PREVIEW_MAX = 1600

// ── 좌표 수학: oriented(θ) ↔ 원본(rot90/flip 후) ──
// renderOriented는 중심 회전 + 외접 확장이므로:
//   src = R(-θ)·(p − C(θ)) + c0,  out = R(θ)·(s − c0) + C(θ)
function baseSize() {
  const swap = props.edits.rot90 % 2 === 1
  const w = bmp?.width ?? 1
  const h = bmp?.height ?? 1
  return { w: swap ? h : w, h: swap ? w : h }
}
function canvasSizeAt(deg: number) {
  const b = baseSize()
  const rad = (deg * Math.PI) / 180
  const cos = Math.abs(Math.cos(rad))
  const sin = Math.abs(Math.sin(rad))
  return deg === 0 ? b : { w: Math.round(b.w * cos + b.h * sin), h: Math.round(b.w * sin + b.h * cos) }
}
function srcOf(p: { x: number; y: number }, deg: number) {
  const C = canvasSizeAt(deg)
  const b = baseSize()
  const rad = (-deg * Math.PI) / 180
  const dx = p.x - C.w / 2
  const dy = p.y - C.h / 2
  return {
    x: dx * Math.cos(rad) - dy * Math.sin(rad) + b.w / 2,
    y: dx * Math.sin(rad) + dy * Math.cos(rad) + b.h / 2,
  }
}
function outOf(s: { x: number; y: number }, deg: number) {
  const C = canvasSizeAt(deg)
  const b = baseSize()
  const rad = (deg * Math.PI) / 180
  const dx = s.x - b.w / 2
  const dy = s.y - b.h / 2
  return {
    x: dx * Math.cos(rad) - dy * Math.sin(rad) + C.w / 2,
    y: dx * Math.sin(rad) + dy * Math.cos(rad) + C.h / 2,
  }
}

/** 미리보기 → 풀해상 배율 */
function fullScale() {
  const os = orientedSize(naturalW, naturalH, { ...props.edits, crop: null })
  return os.w / canvasSizeAt(props.edits.fineDeg).w
}

async function loadPreview() {
  const revision = ++loadRevision
  try {
  const full = await createImageBitmap(props.photo.file, { imageOrientation: 'from-image' })
  if (revision !== loadRevision || disposed) { full.close(); return }
  bmp?.close()
  naturalW = full.width
  naturalH = full.height
  const s = PREVIEW_MAX / Math.max(full.width, full.height)
  if (s < 1) {
    const resized = await createImageBitmap(full, {
      resizeWidth: Math.round(full.width * s),
      resizeHeight: Math.round(full.height * s),
    })
    full.close()
    if (revision !== loadRevision || disposed) { resized.close(); return }
    bmp = resized
  } else {
    bmp = full
  }
  baseKey = '' // 사진 교체 → 베이스 캐시 무효화
  baseCanvas = null
  fitView()
  syncRectFromEdits()
  render()
  } catch (error: any) { emit('error', error.message ?? '사진을 표시하지 못했습니다.') }
}

// oriented 베이스 캐시: 기하(rot90/flip/fine)가 같으면 회전 재계산·픽셀 재독출 생략
// → 밝기/대비 드래그 시 LUT 매핑 + putImageData만 수행 (저사양 PC 렉 완화)
let baseCanvas: HTMLCanvasElement | null = null
let baseData: ImageData | null = null
let outData: ImageData | null = null
let baseKey = ''

function orientedBase(): HTMLCanvasElement {
  const e = props.edits
  const key = `${e.rot90}|${e.flipH}|${e.flipV}|${e.fineDeg}`
  if (!baseCanvas || key !== baseKey) {
    baseCanvas = renderOriented(bmp!, { ...e, crop: null })
    baseData = baseCanvas.getContext('2d')!.getImageData(0, 0, baseCanvas.width, baseCanvas.height)
    outData = new ImageData(baseCanvas.width, baseCanvas.height)
    baseKey = key
  }
  return baseCanvas
}

function render() {
  if (!bmp || !canvasEl.value) return
  // 원본 보기: 기하(회전·플립·크롭)는 유지, 밝기/대비만 보정 전으로
  const e = props.edits
  const c = orientedBase() // 크롭은 오버레이로 표현 — 바깥이 계속 보이도록
  const cv = canvasEl.value
  cv.width = c.width
  cv.height = c.height
  const g = cv.getContext('2d')!
  if (!props.showOriginal && (e.brightness !== 0 || e.contrast !== 0)) {
    const lut = buildLUT(e.brightness, e.contrast)
    const s = baseData!.data
    const d = outData!.data
    for (let i = 0; i < s.length; i += 4) {
      d[i] = lut[s[i]]
      d[i + 1] = lut[s[i + 1]]
      d[i + 2] = lut[s[i + 2]]
      d[i + 3] = s[i + 3]
    }
    g.putImageData(outData!, 0, 0)
  } else {
    g.drawImage(c, 0, 0)
  }
  // view 모드 + 크롭 존재: 바깥 어둡게 (라이트룸식 — 잘린 부분도 계속 보임)
  if (props.mode === 'view' && e.crop) {
    const s = 1 / fullScale()
    dimOutside(g, cv, e.crop.x * s, e.crop.y * s, e.crop.w * s, e.crop.h * s, 0.72)
  }
}

function dimOutside(g: CanvasRenderingContext2D, cv: HTMLCanvasElement,
                    x: number, y: number, w: number, h: number, alpha: number) {
  g.save()
  g.fillStyle = `rgba(10,11,14,${alpha})`
  g.beginPath()
  g.rect(0, 0, cv.width, cv.height)
  g.rect(x, y, w, h)
  g.fill('evenodd')
  g.restore()
}

// ── 크롭 rect ↔ edits.crop 동기화 ──
function syncRectFromEdits() {
  if (!bmp) return
  const C = canvasSizeAt(props.edits.fineDeg)
  if (props.edits.crop) {
    const s = 1 / fullScale()
    const c = props.edits.crop
    rect.value = { x: c.x * s, y: c.y * s, w: c.w * s, h: c.h * s }
  } else if (props.cropRatio) {
    // 비율이 이미 선택돼 있으면 초기 창부터 그 비율로 (중앙)
    let w = C.w * 0.9
    let h = w / props.cropRatio
    if (h > C.h * 0.9) { h = C.h * 0.9; w = h * props.cropRatio }
    rect.value = { x: (C.w - w) / 2, y: (C.h - h) / 2, w, h }
  } else {
    rect.value = { x: C.w * 0.05, y: C.h * 0.05, w: C.w * 0.9, h: C.h * 0.9 }
  }
}

/** 현재 rect를 풀해상 좌표로 (App이 적용 시 호출) */
function currentCrop() {
  const s = fullScale()
  return {
    x: Math.max(0, Math.round(rect.value.x * s)),
    y: Math.max(0, Math.round(rect.value.y * s)),
    w: Math.round(rect.value.w * s),
    h: Math.round(rect.value.h * s),
  }
}
/** 슬라이더/선긋기 미세회전 시: 크롭 중심의 원본 지점을 고정한 새 crop 좌표(풀해상) 계산 */
function cropForFineDeg(newDeg: number): { x: number; y: number; w: number; h: number } | null {
  const c = props.edits.crop
  if (!c || !bmp) return null
  const s = fullScale() // 프리뷰→풀해상 (각도 무관 상수)
  const p1 = { x: (c.x + c.w / 2) / s, y: (c.y + c.h / 2) / s }
  const src = srcOf(p1, props.edits.fineDeg)
  const p2 = outOf(src, newDeg)
  return {
    x: Math.round(p2.x * s - c.w / 2),
    y: Math.round(p2.y * s - c.h / 2),
    w: c.w, h: c.h,
  }
}

defineExpose({ fitView, currentCrop, syncRectFromEdits, cropForFineDeg })

// 비율 변경 시 즉시 반영 — 크롭창 중심 고정, 너비 유지하고 높이 재계산
watch(() => props.cropRatio, (r) => {
  if (props.mode !== 'crop' || !r) return
  const C = canvasSizeAt(props.edits.fineDeg)
  const cx = rect.value.x + rect.value.w / 2
  const cy = rect.value.y + rect.value.h / 2
  let w = rect.value.w
  let h = w / r
  const maxH = C.h * 0.98
  const maxW = C.w * 0.98
  if (h > maxH) { h = maxH; w = h * r }
  if (w > maxW) { w = maxW; h = w / r }
  rect.value = { x: cx - w / 2, y: cy - h / 2, w, h }
})

// 크롭 모드에서 외부(슬라이더)로 fineDeg가 바뀌면 rect도 앵커 유지
watch(() => props.edits.fineDeg, (nv, ov) => {
  if (props.mode !== 'crop' || drag?.kind === 'rotate' || nv === ov) return
  const cx = rect.value.x + rect.value.w / 2
  const cy = rect.value.y + rect.value.h / 2
  const src = srcOf({ x: cx, y: cy }, ov)
  const p2 = outOf(src, nv)
  rect.value.x = p2.x - rect.value.w / 2
  rect.value.y = p2.y - rect.value.h / 2
})

// ── 포인터 인터랙션 ──
type Drag =
  | { kind: 'pan'; x: number; y: number }
  | { kind: 'line' }
  | { kind: 'move'; ox: number; oy: number }
  | { kind: 'resize'; handle: string; anchor: { x: number; y: number } }
  | { kind: 'rotate'; theta0: number; a0: number; s0: { x: number; y: number }; w: number; h: number }
let drag: Drag | null = null

function canvasPoint(ev: MouseEvent) {
  const r = canvasEl.value!.getBoundingClientRect()
  return {
    x: ((ev.clientX - r.left) / r.width) * canvasEl.value!.width,
    y: ((ev.clientY - r.top) / r.height) * canvasEl.value!.height,
  }
}

function hitHandle(p: { x: number; y: number }): string | null {
  const r = rect.value
  const t = Math.max(14, canvasEl.value!.width * 0.015)
  const xs: Record<string, number> = { w: r.x, e: r.x + r.w, c: r.x + r.w / 2 }
  const ys: Record<string, number> = { n: r.y, s: r.y + r.h, c: r.y + r.h / 2 }
  for (const h of HANDLES) {
    const hx = h.includes('w') ? xs.w : h.includes('e') ? xs.e : xs.c
    const hy = h.includes('n') ? ys.n : h.includes('s') ? ys.s : ys.c
    if (Math.abs(p.x - hx) < t && Math.abs(p.y - hy) < t) return h
  }
  return null
}
const inRect = (p: { x: number; y: number }) => {
  const r = rect.value
  return p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h
}

function onDown(ev: MouseEvent) {
  if (!bmp || !canvasEl.value || (ev.button !== 0 && ev.button !== 1)) return
  rootEl.value?.focus({ preventScroll: true })
  if (space.value || ev.button === 1) { drag = { kind: 'pan', x: ev.clientX, y: ev.clientY }; return }
  const p = canvasPoint(ev)
  if (props.straighten) {
    line.value = { x1: p.x, y1: p.y, x2: p.x, y2: p.y }
    drag = { kind: 'line' }
    return
  }
  if (props.mode === 'crop') {
    const h = hitHandle(p)
    if (h) {
      const r = rect.value
      const ax = h.includes('w') ? r.x + r.w : r.x // 반대편 고정점
      const ay = h.includes('n') ? r.y + r.h : r.y
      drag = { kind: 'resize', handle: h, anchor: { x: ax, y: ay } }
    } else if (inRect(p)) {
      drag = { kind: 'move', ox: p.x - rect.value.x, oy: p.y - rect.value.y }
    } else {
      // 바깥 드래그 = 크롭창 중심 피벗 회전
      const cx = rect.value.x + rect.value.w / 2
      const cy = rect.value.y + rect.value.h / 2
      drag = {
        kind: 'rotate',
        theta0: props.edits.fineDeg,
        a0: Math.atan2(p.y - cy, p.x - cx),
        s0: srcOf({ x: cx, y: cy }, props.edits.fineDeg),
        w: rect.value.w, h: rect.value.h,
      }
    }
  } else {
    drag = { kind: 'pan', x: ev.clientX, y: ev.clientY }
  }
}

function onMove(ev: MouseEvent) {
  if (!drag) {
    if (canvasEl.value) updateCursor(canvasPoint(ev))
    return
  }
  const p = canvasPoint(ev)
  if (drag.kind === 'line' && line.value) {
    line.value = { ...line.value, x2: p.x, y2: p.y }
  } else if (drag.kind === 'pan') {
    panX.value += ev.clientX - drag.x
    panY.value += ev.clientY - drag.y
    drag.x = ev.clientX
    drag.y = ev.clientY
  } else if (drag.kind === 'move') {
    const C = canvasSizeAt(props.edits.fineDeg)
    rect.value.x = Math.min(Math.max(p.x - drag.ox, -rect.value.w * 0.5), C.w - rect.value.w * 0.5)
    rect.value.y = Math.min(Math.max(p.y - drag.oy, -rect.value.h * 0.5), C.h - rect.value.h * 0.5)
  } else if (drag.kind === 'resize') {
    const a = drag.anchor
    const hnd = drag.handle
    const corner = hnd.length === 2
    let w = rect.value.w
    let h = rect.value.h
    if (corner) {
      w = Math.abs(p.x - a.x)
      h = Math.abs(p.y - a.y)
      if (props.cropRatio) {
        if (w / h > props.cropRatio) h = w / props.cropRatio
        else w = h * props.cropRatio
      }
    } else if (hnd === 'e' || hnd === 'w') {
      w = Math.abs(p.x - a.x)
      if (props.cropRatio) h = w / props.cropRatio
    } else {
      h = Math.abs(p.y - a.y)
      if (props.cropRatio) w = h * props.cropRatio
    }
    w = Math.max(24, w)
    h = Math.max(24, h)
    let x = rect.value.x
    let y = rect.value.y
    if (corner) {
      x = hnd.includes('w') ? a.x - w : a.x
      y = hnd.includes('n') ? a.y - h : a.y
    } else if (hnd === 'w') { x = a.x - w; if (props.cropRatio) y += (rect.value.h - h) / 2 }
    else if (hnd === 'e') { x = a.x; if (props.cropRatio) y += (rect.value.h - h) / 2 }
    else if (hnd === 'n') { y = a.y - h; if (props.cropRatio) x += (rect.value.w - w) / 2 }
    else { y = a.y; if (props.cropRatio) x += (rect.value.w - w) / 2 }
    rect.value = { x, y, w, h }
  } else if (drag.kind === 'rotate') {
    const cx = rect.value.x + rect.value.w / 2
    const cy = rect.value.y + rect.value.h / 2
    const a = Math.atan2(p.y - cy, p.x - cx)
    let deg = drag.theta0 + ((a - drag.a0) * 180) / Math.PI
    deg = Math.max(-15, Math.min(15, Math.round(deg * 10) / 10))
    emit('fineDeg', deg) // App이 edits.fineDeg 갱신 → 재렌더
    // 크롭 중심이 가리키던 원본 지점 고정 (피벗 = 크롭 중심)
    const c2 = outOf(drag.s0, deg)
    rect.value.x = c2.x - drag.w / 2
    rect.value.y = c2.y - drag.h / 2
  }
}

function onUp() {
  if (drag?.kind === 'line' && line.value) {
    const { x1, y1, x2, y2 } = line.value
    if (Math.hypot(x2 - x1, y2 - y1) > 15) {
      let deg = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI
      if (deg > 45) deg -= 90
      else if (deg < -45) deg += 90
      if (Math.abs(deg) <= 45) emit('angle', deg)
    }
    line.value = null
  }
  drag = null
}

function onWheel(ev: WheelEvent) {
  ev.preventDefault()
  const f = ev.deltaY < 0 ? 1.15 : 1 / 1.15
  const next = Math.min(12, Math.max(0.2, zoom.value * f))
  const r = rootEl.value!.getBoundingClientRect()
  const x = ev.clientX - r.left - r.width / 2, y = ev.clientY - r.top - r.height / 2
  panX.value = x - (x - panX.value) * next / zoom.value
  panY.value = y - (y - panY.value) * next / zoom.value
  zoom.value = next
}

function fitView() {
  zoom.value = 1
  panX.value = 0
  panY.value = 0
}

// rAF 스로틀 재렌더
let rafPending = false
function renderThrottled() {
  if (rafPending) return
  rafPending = true
  requestAnimationFrame(() => {
    rafPending = false
    render()
  })
}

watch(() => props.photo, loadPreview)
watch(() => props.mode, () => {
  fitView()
  if (props.mode === 'crop') syncRectFromEdits()
  renderThrottled()
})
watch(() => [props.edits.rot90, props.edits.flipH, props.edits.flipV,
             props.edits.fineDeg, props.edits.crop, props.showOriginal,
             props.edits.brightness, props.edits.contrast],
  renderThrottled, { deep: true })

onMounted(() => {
  document.addEventListener('mouseup', onUp)
  observer = new ResizeObserver(([entry]) => { available.value = { w: entry.contentRect.width - 28, h: entry.contentRect.height - 28 } })
  if (rootEl.value) observer.observe(rootEl.value)
  loadPreview()
})
onUnmounted(() => {
  disposed = true; loadRevision++; observer?.disconnect()
  document.removeEventListener('mouseup', onUp)
  bmp?.close()
})

// SVG 오버레이용 (크롭 모드)
function handlePos(h: string) {
  const r = rect.value
  return {
    x: h.includes('w') ? r.x : h.includes('e') ? r.x + r.w : r.x + r.w / 2,
    y: h.includes('n') ? r.y : h.includes('s') ? r.y + r.h : r.y + r.h / 2,
  }
}
</script>

<template>
  <div
    ref="rootEl" tabindex="0" aria-label="사진 편집 캔버스"
    class="editor-canvas"
    :class="{ straighten, cropmode: mode === 'crop' }"
    :style="cursor ? { cursor } : {}"
    @wheel="onWheel"
    @mousedown.prevent="onDown"
    @mousemove="onMove"
    @keydown="(e) => { if (e.code === 'Space' && !e.ctrlKey && !e.metaKey) { e.preventDefault(); space = true } }"
    @keyup="(e) => { if (e.code === 'Space') space = false }"
    @blur="space = false"
  >
    <div class="stage" :style="{ transform: `translate(${panX}px, ${panY}px) scale(${zoom})` }">
      <div class="canvas-holder">
        <canvas ref="canvasEl" :style="{ maxWidth: `${available.w}px`, maxHeight: `${available.h}px` }" />
        <svg v-if="mode === 'crop' && canvasEl" class="crop-overlay"
             :viewBox="`0 0 ${canvasEl.width} ${canvasEl.height}`" preserveAspectRatio="none">
          <path :d="`M0 0 H${canvasEl.width} V${canvasEl.height} H0 Z
                     M${rect.x} ${rect.y} h${rect.w} v${rect.h} h${-rect.w} Z`"
                fill="rgba(10,11,14,0.6)" fill-rule="evenodd" />
          <rect :x="rect.x" :y="rect.y" :width="rect.w" :height="rect.h"
                fill="none" stroke="#fff" stroke-width="1.5" vector-effect="non-scaling-stroke" />
          <line v-for="i in 2" :key="'v' + i" :x1="rect.x + (rect.w * i) / 3" :y1="rect.y"
                :x2="rect.x + (rect.w * i) / 3" :y2="rect.y + rect.h"
                stroke="rgba(255,255,255,0.35)" vector-effect="non-scaling-stroke" />
          <line v-for="i in 2" :key="'h' + i" :x1="rect.x" :y1="rect.y + (rect.h * i) / 3"
                :x2="rect.x + rect.w" :y2="rect.y + (rect.h * i) / 3"
                stroke="rgba(255,255,255,0.35)" vector-effect="non-scaling-stroke" />
          <rect v-for="h in HANDLES" :key="h"
                :x="handlePos(h).x - 7" :y="handlePos(h).y - 7" width="14" height="14"
                fill="#fff" stroke="#4c8dff" stroke-width="1.5" />
        </svg>
        <svg v-if="line && canvasEl" class="line-overlay"
             :viewBox="`0 0 ${canvasEl.width} ${canvasEl.height}`" preserveAspectRatio="none">
          <line :x1="line.x1" :y1="line.y1" :x2="line.x2" :y2="line.y2"
                vector-effect="non-scaling-stroke" />
        </svg>
      </div>
    </div>
    <div v-if="straighten" class="straighten-hint">
      수평(또는 수직)이어야 할 선을 드래그로 그으세요 — 자동으로 각도 보정
    </div>
    <div v-else-if="mode === 'crop'" class="straighten-hint">
      안쪽 드래그=이동 · 꼭지점=크기(비율 유지) · <b>바깥 드래그=회전</b> · Enter=적용
    </div>
  </div>
</template>

<style scoped>
.editor-canvas { width:100%;height:100%;overflow:hidden;position:relative;cursor:grab;outline:none;min-height:150px; }
.editor-canvas:focus-visible { outline:2px solid #63a9ef;outline-offset:-2px }
.stage { width:100%;height:100%;display:flex;align-items:center;justify-content:center;transform-origin:center }
.canvas-holder { position:relative;display:flex;flex-shrink:0 }
canvas { display:block;object-fit:contain;user-select:none }
.crop-overlay,.line-overlay { position:absolute;inset:0;width:100%;height:100%;pointer-events:none }
.line-overlay line { stroke:#63a9ef;stroke-width:2;stroke-dasharray:6 4 }
.straighten-hint { position:absolute;bottom:12px;left:50%;transform:translateX(-50%);background:#12161ee8;color:#d6e8fa;padding:8px 12px;border-radius:6px;font-size:12px;width:max-content;max-width:95%;pointer-events:none }
</style>
