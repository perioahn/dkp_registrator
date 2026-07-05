<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'

const props = defineProps<{
  img: {
    id: string; name: string
    result: null | {
      status: string; gate: string; label: string; reason?: string
      n_inlier?: number; inlier_ratio?: number; reproj_median?: number
      rotation_deg?: number; scale?: number
    }
  }
}>()

const mode = ref<'wipe' | 'false' | 'flicker' | 'side' | 'match'>('wipe')
const wipe = ref(50)
const fcOpacity = ref(70)
const flickOn = ref(true)
let flickTimer: number | undefined

// 동기 pan/zoom
const zoom = ref(1)
const panX = ref(0)
const panY = ref(0)
let dragging = false
let lastX = 0
let lastY = 0

const r = computed(() => props.img.result!)
const fixedUrl = computed(() => `/api/result/${props.img.id}/fixed`)
const regUrl = computed(() => `/api/result/${props.img.id}/registered`)
const fcUrl = computed(() => `/api/result/${props.img.id}/false_color`)
const matchUrl = computed(() => `/api/result/${props.img.id}/match_viz`)

const transform = computed(() =>
  `translate(${panX.value}px, ${panY.value}px) scale(${zoom.value})`)

function onWheel(e: WheelEvent) {
  e.preventDefault()
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
  zoom.value = Math.min(12, Math.max(0.3, zoom.value * factor))
}
function onDown(e: MouseEvent) { dragging = true; lastX = e.clientX; lastY = e.clientY }
function onMove(e: MouseEvent) {
  if (!dragging) return
  panX.value += e.clientX - lastX
  panY.value += e.clientY - lastY
  lastX = e.clientX; lastY = e.clientY
}
function onUp() { dragging = false }
function resetView() { zoom.value = 1; panX.value = 0; panY.value = 0 }

function setMode(m: typeof mode.value) {
  mode.value = m
  if (m === 'flicker') {
    flickTimer = window.setInterval(() => (flickOn.value = !flickOn.value), 600)
  } else if (flickTimer) {
    clearInterval(flickTimer)
    flickTimer = undefined
  }
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

onMounted(() => document.addEventListener('mouseup', onUp))
onUnmounted(() => {
  document.removeEventListener('mouseup', onUp)
  if (flickTimer) clearInterval(flickTimer)
})
</script>

<template>
  <div class="result-viewer">
    <div class="toolbar">
      <span class="badge big" :class="badgeCls">{{ r.status.toUpperCase() }}</span>
      <span class="metrics" v-if="r.status !== 'fail'">
        {{ r.label }} · {{ r.gate }} · inlier {{ r.n_inlier }}
        ({{ ((r.inlier_ratio ?? 0) * 100).toFixed(0) }}%) ·
        reproj {{ r.reproj_median?.toFixed(1) }}px
        <template v-if="r.rotation_deg != null"> · rot {{ r.rotation_deg.toFixed(1) }}°</template>
        <template v-if="r.scale != null"> · scale {{ r.scale.toFixed(3) }}</template>
      </span>
      <span v-else class="metrics fail-reason">{{ failText }}</span>
      <span class="spacer" />
      <a v-if="r.status !== 'fail'" class="dl-btn" :href="`/api/result/${img.id}/download`">💾 저장</a>
    </div>

    <template v-if="r.status !== 'fail'">
      <div class="mode-tabs sub">
        <button :class="{ on: mode === 'wipe' }" @click="setMode('wipe')">와이프</button>
        <button :class="{ on: mode === 'false' }" @click="setMode('false')">False color</button>
        <button :class="{ on: mode === 'flicker' }" @click="setMode('flicker')">플리커</button>
        <button :class="{ on: mode === 'side' }" @click="setMode('side')">나란히</button>
        <button :class="{ on: mode === 'match' }" @click="setMode('match')">매칭점</button>
        <span class="spacer" />
        <label v-if="mode === 'wipe'" class="slider-label">
          기준 ◀ <input v-model.number="wipe" type="range" min="0" max="100" /> ▶ 정합
        </label>
        <label v-if="mode === 'false'" class="slider-label">
          투명도 <input v-model.number="fcOpacity" type="range" min="0" max="100" />
        </label>
        <button class="reset-view" @click="resetView">⟲ 뷰 리셋</button>
      </div>

      <div
        class="viewer-area"
        @wheel="onWheel"
        @mousedown.prevent="onDown"
        @mousemove="onMove"
      >
        <div v-if="mode === 'side'" class="side-by-side">
          <div class="pane"><img :src="fixedUrl" :style="{ transform }" /><span>기준</span></div>
          <div class="pane"><img :src="regUrl" :style="{ transform }" /><span>정합</span></div>
        </div>
        <div v-else-if="mode === 'match'" class="single">
          <img :src="matchUrl" :style="{ transform }" />
        </div>
        <div v-else class="stacked">
          <img :src="fixedUrl" :style="{ transform }" />
          <img
            v-if="mode === 'wipe'"
            :src="regUrl"
            :style="{ transform, clipPath: `inset(0 ${100 - wipe}% 0 0)` }"
          />
          <img v-else-if="mode === 'false'" :src="fcUrl"
               :style="{ transform, opacity: fcOpacity / 100 }" />
          <img v-else-if="mode === 'flicker'" :src="regUrl"
               :style="{ transform, opacity: flickOn ? 1 : 0 }" />
          <div v-if="mode === 'wipe'" class="wipe-line" :style="{ left: wipe + '%' }" />
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
