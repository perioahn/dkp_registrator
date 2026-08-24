<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { fcOpacity, mode, wipe, type ViewMode } from '../viewstate'

const props = defineProps<{
  img: {
    id: string; name: string
    result: null | {
      status: string; gate: string; label: string; reason?: string
      n_inlier?: number; inlier_ratio?: number; reproj_median?: number
      rotation_deg?: number; scale?: number; manual_adjusted?: boolean
      used_mask?: boolean
    }
  }
}>()
const emit = defineEmits<{ changed: [] }>()

// mode/wipe/fcOpacity는 viewstate 모듈 — 다른 Moving을 선택해도 보기 방식 유지
const flickOn = ref(true)
let flickTimer: number | undefined

// 동기 pan/zoom
const zoom = ref(1)
const panX = ref(0)
const panY = ref(0)
let dragging = false
let lastX = 0
let lastY = 0

const stackedEl = ref<HTMLElement | null>(null)
const baseImgEl = ref<HTMLImageElement | null>(null)
// 와이프 경계선: 이미지 표시 영역 기준 (컨테이너 아님 — 경계와 속도 일치)
const lineBox = ref({ left: 0, top: 0, height: 0 })
// 수동 미세조정 오버레이: 이미지 표시 영역 rect (stacked 좌표)
const imgBox = ref({ left: 0, top: 0, w: 0, h: 0 })

const r = computed(() => props.img.result!)
const ver = ref(0) // 미세조정 적용 후 이미지 강제 리로드
const fixedUrl = computed(() => `/api/result/${props.img.id}/fixed`)
const regUrl = computed(() => `/api/result/${props.img.id}/registered?v=${ver.value}`)
const fcUrl = computed(() => `/api/result/${props.img.id}/false_color?v=${ver.value}`)
// src가 바뀌면 디코드 완료까지 숨김 — 부분 렌더 순간 노출 방지
const regLoaded = ref(false)
const fcLoaded = ref(false)
watch([regUrl], () => { regLoaded.value = false })
watch([fcUrl], () => { fcLoaded.value = false })
const matchUrl = computed(() => `/api/result/${props.img.id}/match_viz`)

const transform = computed(() =>
  `translate(${panX.value}px, ${panY.value}px) scale(${zoom.value})`)

// ── 수동 미세조정 (dx/dy: zoom=1 표시 px, 중심 기준 rot/scale) ──
const adj = ref({ dx: 0, dy: 0, rot: 0, s: 1 })
const applying = ref(false)
const adjDirty = computed(() =>
  adj.value.dx !== 0 || adj.value.dy !== 0 || adj.value.rot !== 0 || adj.value.s !== 1)
type AdjDrag =
  | { kind: 'move'; x: number; y: number }
  | { kind: 'scale'; d0: number; s0: number }
  | { kind: 'rotate'; a0: number; rot0: number }
let adjDrag: AdjDrag | null = null
const CORNERS = ['nw', 'ne', 'se', 'sw'] as const
const ROTATE_CURSOR = `url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='22' height='22' viewBox='0 0 24 24'><path d='M12 4a8 8 0 1 1-7.5 5.3' fill='none' stroke='black' stroke-width='4.5' stroke-linecap='round'/><path d='M12 4a8 8 0 1 1-7.5 5.3' fill='none' stroke='white' stroke-width='2.5' stroke-linecap='round'/><path d='M2.5 3.5 L5 10 L11 7 Z' fill='white' stroke='black' stroke-width='1'/></svg>") 11 11, grab`

const ghostTransform = computed(() =>
  `${transform.value} translate(${adj.value.dx}px, ${adj.value.dy}px)` +
  ` rotate(${adj.value.rot}deg) scale(${adj.value.s})`)
// 오버레이는 stacked 좌표(줌 반영됨) — delta 이동만 줌 배율로
const overlayTransform = computed(() =>
  `translate(${adj.value.dx * zoom.value}px, ${adj.value.dy * zoom.value}px)` +
  ` rotate(${adj.value.rot}deg) scale(${adj.value.s})`)

function ghostCenter() {
  const ir = baseImgEl.value!.getBoundingClientRect()
  return {
    x: ir.left + ir.width / 2 + adj.value.dx * zoom.value,
    y: ir.top + ir.height / 2 + adj.value.dy * zoom.value,
  }
}
function onAdjMoveDown(e: MouseEvent) {
  adjDrag = { kind: 'move', x: e.clientX, y: e.clientY }
}
function onAdjHandleDown(e: MouseEvent) {
  const c = ghostCenter()
  adjDrag = { kind: 'scale', d0: Math.hypot(e.clientX - c.x, e.clientY - c.y), s0: adj.value.s }
}
async function applyAdjust(reset = false) {
  const img = baseImgEl.value
  if (!img || applying.value) return
  let body: Record<string, unknown> = { reset: true }
  if (!reset) {
    if (!adjDirty.value) return
    // 표시 px → 서빙 이미지 px (dx/dy는 zoom=1 기준이라 rect/zoom이 표시폭)
    const k = img.naturalWidth / (img.getBoundingClientRect().width / zoom.value)
    body = {
      dx: adj.value.dx * k, dy: adj.value.dy * k,
      scale: adj.value.s, rot_deg: adj.value.rot, ref_w: img.naturalWidth,
    }
  }
  applying.value = true
  const res = await fetch(`/api/result/${props.img.id}/adjust`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  applying.value = false
  if (!res.ok) return
  adj.value = { dx: 0, dy: 0, rot: 0, s: 1 }
  ver.value = Date.now()
  emit('changed')
}

function updateLine() {
  const img = baseImgEl.value
  const box = stackedEl.value
  if (!img || !box) return
  const ir = img.getBoundingClientRect()
  const br = box.getBoundingClientRect()
  lineBox.value = {
    left: ir.left - br.left + (wipe.value / 100) * ir.width,
    top: ir.top - br.top,
    height: ir.height,
  }
  imgBox.value = {
    left: ir.left - br.left, top: ir.top - br.top,
    w: ir.width, h: ir.height,
  }
}

function onWheel(e: WheelEvent) {
  e.preventDefault()
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
  zoom.value = Math.min(12, Math.max(0.3, zoom.value * factor))
}
function onDown(e: MouseEvent) {
  if (mode.value === 'adjust') {
    // 바깥 드래그 = 이동상 중심 피벗 회전 (안쪽/핸들은 stop이라 여기 안 옴)
    const c = ghostCenter()
    adjDrag = { kind: 'rotate', a0: Math.atan2(e.clientY - c.y, e.clientX - c.x), rot0: adj.value.rot }
    return
  }
  dragging = true; lastX = e.clientX; lastY = e.clientY
}
function onMove(e: MouseEvent) {
  if (adjDrag) {
    if (adjDrag.kind === 'move') {
      adj.value.dx += (e.clientX - adjDrag.x) / zoom.value
      adj.value.dy += (e.clientY - adjDrag.y) / zoom.value
      adjDrag.x = e.clientX; adjDrag.y = e.clientY
    } else if (adjDrag.kind === 'scale') {
      const c = ghostCenter()
      const d = Math.hypot(e.clientX - c.x, e.clientY - c.y)
      adj.value.s = Math.min(3, Math.max(0.3, adjDrag.s0 * (d / adjDrag.d0)))
    } else {
      const c = ghostCenter()
      const a = Math.atan2(e.clientY - c.y, e.clientX - c.x)
      adj.value.rot = adjDrag.rot0 + ((a - adjDrag.a0) * 180) / Math.PI
    }
    return
  }
  if (wipeDragging) {
    const ir = baseImgEl.value?.getBoundingClientRect()
    if (ir?.width) wipe.value = Math.min(100, Math.max(0, ((e.clientX - ir.left) / ir.width) * 100))
    return
  }
  if (!dragging) return
  panX.value += e.clientX - lastX
  panY.value += e.clientY - lastY
  lastX = e.clientX; lastY = e.clientY
}
function onUp() { dragging = false; wipeDragging = false; adjDrag = null }

// 와이프 선 가운데 핸들 드래그
let wipeDragging = false
function onHandleDown() { wipeDragging = true }
function resetView() { zoom.value = 1; panX.value = 0; panY.value = 0 }

function stopFlicker() {
  if (flickTimer) {
    clearInterval(flickTimer)
    flickTimer = undefined
  }
}
function startFlicker() {
  stopFlicker()
  flickTimer = window.setInterval(() => (flickOn.value = !flickOn.value), 600)
}

function setMode(m: ViewMode) {
  if (mode.value === 'adjust' && m !== 'adjust') adj.value = { dx: 0, dy: 0, rot: 0, s: 1 }
  mode.value = m
  if (m === 'flicker') startFlicker()
  else stopFlicker()
}

const badgeCls = computed(() =>
  r.value.status === 'pass' ? 'ok' : r.value.status === 'warn' ? 'warn' : 'fail')

const failText = computed(() => {
  if (r.value.status !== 'fail') return ''
  const reason = r.value.reason ?? ''
  if (reason.includes('matches')) return '특징점 매칭 부족 — 마스크를 더 넓게 지정해 보세요'
  if (reason.includes('gate')) return '품질 게이트 실패 — 마스크 재지정 또는 relaxed 프로필 시도'
  return reason
})

watch([wipe, zoom, panX, panY, mode], () => nextTick(updateLine))

onMounted(() => {
  document.addEventListener('mouseup', onUp)
  window.addEventListener('resize', updateLine)
  if (mode.value === 'flicker') startFlicker() // 유지된 모드 복원
  nextTick(updateLine)
})
onUnmounted(() => {
  document.removeEventListener('mouseup', onUp)
  window.removeEventListener('resize', updateLine)
  stopFlicker()
})
</script>

<template>
  <div class="result-viewer">
    <div class="toolbar">
      <span class="badge big" :class="badgeCls">{{ r.status.toUpperCase() }}</span>
      <span v-if="r.manual_adjusted" class="badge warn">수동조정</span>
      <span class="badge" :class="r.used_mask ? 'ok' : 'plain'">
        {{ r.used_mask ? '마스크 정합' : '전체영역 정합' }}
      </span>
      <span class="metrics" v-if="r.status !== 'fail'">
        {{ r.label }} · {{ r.gate }} · inlier {{ r.n_inlier }}
        ({{ ((r.inlier_ratio ?? 0) * 100).toFixed(0) }}%) ·
        reproj {{ r.reproj_median?.toFixed(1) }}px
        <template v-if="r.rotation_deg != null"> · rot {{ r.rotation_deg.toFixed(1) }}°</template>
        <template v-if="r.scale != null"> · scale {{ r.scale.toFixed(3) }}</template>
        <template v-if="r.manual_adjusted"> · (지표는 자동정합 기준)</template>
      </span>
      <span v-else class="metrics fail-reason">{{ failText }}</span>
      <span class="spacer" />
    </div>

    <template v-if="r.status !== 'fail'">
      <div class="mode-tabs sub">
        <button :class="{ on: mode === 'wipe' }" @click="setMode('wipe')">Wipe</button>
        <button :class="{ on: mode === 'false' }" @click="setMode('false')">False color</button>
        <button :class="{ on: mode === 'flicker' }" @click="setMode('flicker')">Flicker</button>
        <button :class="{ on: mode === 'side' }" @click="setMode('side')">Side by side</button>
        <button :class="{ on: mode === 'match' }" @click="setMode('match')">Matches</button>
        <button :class="{ on: mode === 'adjust' }" @click="setMode('adjust')">수동 미세조정</button>
        <span class="spacer" />
        <label v-if="mode === 'wipe'" class="slider-label">
          Fixed ◀ <input v-model.number="wipe" type="range" min="0" max="100" /> ▶ Moving
        </label>
        <label v-if="mode === 'false'" class="slider-label">
          투명도 <input v-model.number="fcOpacity" type="range" min="0" max="100" />
        </label>
        <template v-if="mode === 'adjust'">
          <span class="adj-hint">안쪽 드래그=이동 · 꼭지점=크기 · <b>바깥 드래그=회전</b></span>
          <button class="adj-apply" :disabled="applying || !adjDirty" @click="applyAdjust()">
            {{ applying ? '적용 중…' : '✓ 적용' }}
          </button>
          <button :disabled="applying" @click="applyAdjust(true)">↺ 자동정합 복원</button>
        </template>
        <button class="reset-view" @click="resetView">⟲ 뷰 리셋</button>
      </div>

      <div
        class="viewer-area"
        :style="mode === 'adjust' ? { cursor: ROTATE_CURSOR } : {}"
        @wheel="onWheel"
        @mousedown.prevent="onDown"
        @mousemove="onMove"
      >
        <div v-if="mode === 'side'" class="side-by-side">
          <div class="pane"><img :src="fixedUrl" :style="{ transform }" /><span>Fixed</span></div>
          <div class="pane"><img :src="regUrl" :style="{ transform }" /><span>Moving (registered)</span></div>
        </div>
        <div v-else-if="mode === 'match'" class="single">
          <img :src="matchUrl" :style="{ transform }" />
        </div>
        <div v-else ref="stackedEl" class="stacked">
          <img ref="baseImgEl" :src="fixedUrl" :style="{ transform }" @load="updateLine" />
          <!-- 디코드 완료 전 표시 금지 (visibility) — 부분 렌더/옛 프레임이 순간 왜곡처럼 보이는 것 방지 -->
          <img
            v-if="mode === 'wipe'"
            :src="regUrl"
            :style="{ transform, clipPath: `inset(0 ${100 - wipe}% 0 0)`,
                      visibility: regLoaded ? 'visible' : 'hidden' }"
            @load="regLoaded = true"
          />
          <img v-else-if="mode === 'false'" :src="fcUrl"
               :style="{ transform, opacity: fcOpacity / 100,
                         visibility: fcLoaded ? 'visible' : 'hidden' }"
               @load="fcLoaded = true" />
          <img v-else-if="mode === 'flicker'" :src="regUrl"
               :style="{ transform, opacity: flickOn && regLoaded ? 1 : 0 }"
               @load="regLoaded = true" />
          <img v-else-if="mode === 'adjust'" :src="regUrl"
               :style="{ transform: ghostTransform,
                         opacity: regLoaded ? 0.5 : 0 }"
               @load="regLoaded = true" />
          <div v-if="mode === 'wipe'" class="wipe-line"
               :style="{ left: lineBox.left + 'px', top: lineBox.top + 'px',
                         height: lineBox.height + 'px' }">
            <div class="wipe-handle" @mousedown.stop.prevent="onHandleDown">◀ ▶</div>
          </div>
          <div v-if="mode === 'adjust'" class="adj-rect"
               :style="{ left: imgBox.left + 'px', top: imgBox.top + 'px',
                         width: imgBox.w + 'px', height: imgBox.h + 'px',
                         transform: overlayTransform }"
               @mousedown.stop.prevent="onAdjMoveDown">
            <div v-for="c in CORNERS" :key="c" class="adj-handle" :class="c"
                 @mousedown.stop.prevent="onAdjHandleDown" />
          </div>
        </div>
      </div>
    </template>
    <div v-else class="fail-help">
      <p>이 이미지의 정합이 실패했습니다.</p>
      <ul>
        <li>마스크가 치아 영역을 충분히 덮는지 확인</li>
        <li>사진 방향이 크게 다르면 <b>Lazy 모드</b>로 재실행</li>
        <li>relaxed 프로필로 재시도</li>
      </ul>
    </div>
  </div>
</template>
