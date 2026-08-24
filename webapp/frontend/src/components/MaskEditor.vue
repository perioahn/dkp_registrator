<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'

const props = defineProps<{
  img: { id: string; name: string; w: number; h: number; n_objects: number }
  fixed?: { id: string; name: string; mask_ready: boolean; mask_rev: number } | null
}>()
const emit = defineEmits<{ changed: []; gotoFixed: [] }>()

// 고정상 마스크 미니맵: 이동상 작업 중 + 고정상에 마스크가 있을 때만
const showMinimap = computed(() =>
  !!props.fixed && props.fixed.id !== props.img.id && props.fixed.mask_ready)
const miniOpen = ref(true)
const miniBig = ref(false)

const overlayTs = ref(0)
const busy = ref(false)
const nObjects = ref(props.img.n_objects)
const wrap = ref<HTMLElement | null>(null)

async function clickAt(ev: MouseEvent, label: number) {
  if (busy.value) return
  const el = ev.currentTarget as HTMLElement
  const rect = el.getBoundingClientRect()
  // 표시 크기 → work 좌표 변환
  const x = ((ev.clientX - rect.left) / rect.width) * props.img.w
  const y = ((ev.clientY - rect.top) / rect.height) * props.img.h
  busy.value = true
  try {
    const res = await fetch(`/api/mask/${props.img.id}/click`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x, y, label }),
    })
    if (res.ok) overlayTs.value = (await res.json()).ts
  } finally {
    busy.value = false
  }
  emit('changed')
}

async function action(a: 'confirm' | 'undo' | 'reset') {
  busy.value = true
  try {
    const res = await fetch(`/api/mask/${props.img.id}/action`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: a }),
    })
    if (res.ok) {
      const d = await res.json()
      nObjects.value = d.n_objects
      overlayTs.value = d.ts
    }
  } finally {
    busy.value = false
  }
  emit('changed')
}

function onKey(e: KeyboardEvent) {
  if (e.key === 'z' || e.key === 'Z') action('confirm')
  else if (e.key === 'x' || e.key === 'X') action('reset')
}

onMounted(() => window.addEventListener('keydown', onKey))
onUnmounted(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <div class="mask-editor">
    <div class="toolbar">
      <span class="tip">좌클릭 = 포함 · 우클릭 = 제외 · 개체 확정 후 다음 개체 진행</span>
      <span class="spacer" />
      <span v-if="nObjects > 0" class="obj-count">마스크 ✓</span>
      <button :disabled="busy" @click="action('confirm')">✓ 개체 확정 (Z)</button>
      <button :disabled="busy" @click="action('undo')">↶ 실행취소</button>
      <button :disabled="busy" class="danger" @click="action('reset')">✕ 초기화 (X)</button>
    </div>
    <div ref="wrap" class="canvas-wrap">
      <div
        class="canvas-stack"
        :class="{ busy }"
        @click="clickAt($event, 1)"
        @contextmenu.prevent="clickAt($event, 0)"
      >
        <img class="base" :src="`/api/image/${img.id}`" draggable="false" />
        <img class="overlay" :src="`/api/mask/${img.id}/overlay?t=${overlayTs}`" draggable="false" />
      </div>
      <!-- 고정상 마스크 미니맵 — 이동상 마스킹 시 기준 마스크 위치 참고용 -->
      <div v-if="showMinimap" class="fixed-minimap" :class="{ big: miniBig, closed: !miniOpen }">
        <div class="fm-head">
          <span>기준 마스크</span>
          <button class="fm-btn" @click="emit('gotoFixed')" title="기준 이미지 마스크 수정으로 이동">수정</button>
          <button class="fm-btn" @click="miniOpen = !miniOpen">{{ miniOpen ? '▾' : '▸' }}</button>
        </div>
        <div v-if="miniOpen" class="fm-stack" title="클릭 = 확대/축소" @click="miniBig = !miniBig">
          <img :src="`/api/image/${fixed!.id}`" draggable="false" />
          <img class="fm-overlay" :src="`/api/mask/${fixed!.id}/overlay?t=${fixed!.mask_rev}`" draggable="false" />
        </div>
      </div>
    </div>
  </div>
</template>
