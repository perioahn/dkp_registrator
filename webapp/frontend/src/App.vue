<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import MaskEditor from './components/MaskEditor.vue'
import ResultViewer from './components/ResultViewer.vue'

interface ImgInfo {
  id: string; role: 'fixed' | 'moving'; name: string
  w: number; h: number; n_objects: number; has_current: boolean
  mask_ready: boolean
  result: null | {
    status: string; gate: string; label: string; reason?: string
    n_inlier?: number; inlier_ratio?: number; reproj_median?: number
    rotation_deg?: number; scale?: number
  }
}

const images = ref<ImgInfo[]>([])
const running = ref(false)
const profiles = ref<string[]>(['normal'])
const selectedId = ref<string | null>(null)
const viewMode = ref<'mask' | 'result'>('mask')
const lazy = ref(false)
const profile = ref('normal')
const msg = ref('')
let es: EventSource | null = null

const selected = computed(() => images.value.find((i) => i.id === selectedId.value) ?? null)
const fixedImg = computed(() => images.value.find((i) => i.role === 'fixed') ?? null)
const canRegister = computed(() =>
  !running.value && fixedImg.value?.mask_ready &&
  images.value.some((i) => i.role === 'moving' && i.mask_ready))

async function refresh() {
  const d = await (await fetch('/api/state')).json()
  images.value = d.images
  running.value = d.running
  profiles.value = d.profiles
  if (!selectedId.value && d.images.length) selectedId.value = d.images[0].id
  if (selectedId.value && !d.images.some((i: ImgInfo) => i.id === selectedId.value)) {
    selectedId.value = d.images[0]?.id ?? null
  }
}

async function uploadFiles(role: 'fixed' | 'moving', ev: Event) {
  const input = ev.target as HTMLInputElement
  if (!input.files?.length) return
  const fd = new FormData()
  for (const f of input.files) fd.append('files', f)
  const res = await fetch(`/api/upload?role=${role}`, { method: 'POST', body: fd })
  if (!res.ok) msg.value = (await res.json()).detail ?? '업로드 실패'
  else {
    const d = await res.json()
    await refresh()
    if (d.added.length) { selectedId.value = d.added[0]; viewMode.value = 'mask' }
  }
  input.value = ''
}

async function removeImage(id: string) {
  await fetch(`/api/image/${id}/delete`, { method: 'POST' })
  await refresh()
}

async function startRegister() {
  msg.value = ''
  const res = await fetch('/api/register', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lazy: lazy.value, profile: profile.value }),
  })
  const d = await res.json()
  if (!res.ok) { msg.value = d.detail ?? '실행 실패'; return }
  running.value = true
  if (d.skipped_no_mask?.length) {
    msg.value = `마스크 없는 ${d.skipped_no_mask.length}장 제외`
  }
}

function select(id: string, mode: 'mask' | 'result' = 'mask') {
  selectedId.value = id
  viewMode.value = mode
}

function badge(r: ImgInfo['result']): { text: string; cls: string } | null {
  if (!r) return null
  if (r.status === 'pass') return { text: 'PASS', cls: 'ok' }
  if (r.status === 'warn') return { text: 'WARN', cls: 'warn' }
  return { text: 'FAIL', cls: 'fail' }
}

onMounted(() => {
  refresh()
  es = new EventSource('/api/events')
  es.addEventListener('register', (e) => {
    const d = JSON.parse((e as MessageEvent).data)
    if (d.state === 'progress') msg.value = `정합 중 ${d.done + 1}/${d.total}: ${d.name}`
    else if (d.state === 'lazy') msg.value = `Lazy ${d.lazy_cur}/${d.lazy_total} (${d.lazy_label}) — ${d.done + 1}/${d.total}`
    else if (d.state === 'one_done') refresh()
    else if (d.state === 'done') {
      msg.value = `정합 완료 (${d.total}장)`
      running.value = false
      refresh()
      const firstMoving = images.value.find((i) => i.role === 'moving' && i.result)
      if (firstMoving) select(firstMoving.id, 'result')
    } else if (d.state === 'error') {
      msg.value = `오류: ${d.detail}`
      running.value = false
    }
  })
})
onUnmounted(() => es?.close())
</script>

<template>
  <div class="layout">
    <aside class="sidebar">
      <h1>DKP Registrator</h1>

      <div class="upload-row">
        <label class="up-btn fixed-btn">
          기준(Fixed) 선택
          <input type="file" accept="image/*" hidden @change="uploadFiles('fixed', $event)" />
        </label>
        <label class="up-btn">
          + Moving 추가
          <input type="file" accept="image/*" multiple hidden @change="uploadFiles('moving', $event)" />
        </label>
      </div>

      <div class="img-list">
        <div
          v-for="img in images"
          :key="img.id"
          class="img-item"
          :class="{ active: img.id === selectedId, fixed: img.role === 'fixed' }"
          @click="select(img.id, img.result ? 'result' : 'mask')"
        >
          <img :src="`/api/image/${img.id}`" />
          <div class="img-meta">
            <div class="img-name">
              <span v-if="img.role === 'fixed'" class="role-tag">기준</span>{{ img.name }}
            </div>
            <div class="img-sub">
              <span :class="{ 'mask-ok': img.mask_ready }">
                {{ img.mask_ready ? `마스크 ${img.n_objects || '작업중'}` : '마스크 필요' }}
              </span>
              <span v-if="badge(img.result)" class="badge" :class="badge(img.result)!.cls">
                {{ badge(img.result)!.text }}
              </span>
            </div>
          </div>
          <button class="rm" @click.stop="removeImage(img.id)">×</button>
        </div>
        <p v-if="!images.length" class="hint">
          기준 사진 1장과 정합할 사진들을 올리고,<br />
          각 사진에서 치아 영역을 클릭해 마스크를 만드세요.
        </p>
      </div>

      <div class="run-panel">
        <label class="chk"><input v-model="lazy" type="checkbox" /> Lazy (방향 자동탐색)</label>
        <label class="chk">프로필
          <select v-model="profile">
            <option v-for="p in profiles" :key="p" :value="p">{{ p }}</option>
          </select>
        </label>
        <button class="register-btn" :disabled="!canRegister" @click="startRegister">
          {{ running ? '정합 중…' : '▶ Register' }}
        </button>
        <div class="statusmsg">{{ msg }}</div>
      </div>
    </aside>

    <main class="content">
      <template v-if="selected">
        <div class="mode-tabs">
          <button :class="{ on: viewMode === 'mask' }" @click="viewMode = 'mask'">마스크</button>
          <button :class="{ on: viewMode === 'result' }" :disabled="!selected.result"
                  @click="viewMode = 'result'">결과</button>
        </div>
        <MaskEditor v-if="viewMode === 'mask'" :key="selected.id" :img="selected" @changed="refresh" />
        <ResultViewer v-else-if="selected.result" :key="'r' + selected.id" :img="selected" />
      </template>
      <p v-else class="hint center">← 이미지를 업로드하세요</p>
    </main>
  </div>
</template>
