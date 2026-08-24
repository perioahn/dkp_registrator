<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import MaskEditor from './components/MaskEditor.vue'
import ResultViewer from './components/ResultViewer.vue'

interface ImgInfo {
  id: string; role: 'fixed' | 'moving'; name: string
  w: number; h: number; n_objects: number; has_current: boolean
  mask_ready: boolean; mask_rev: number
  result: null | {
    status: string; gate: string; label: string; reason?: string
    n_inlier?: number; inlier_ratio?: number; reproj_median?: number
    rotation_deg?: number; scale?: number; manual_adjusted?: boolean
    used_mask?: boolean
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
// 마스크는 선택 사항 — 기준 + Moving만 있으면 실행 가능 (페어 양쪽 마스크 시에만 마스크 정합)
const canRegisterAll = computed(() =>
  !running.value && !!fixedImg.value && images.value.some((i) => i.role === 'moving'))
// 선택 정합: 체크된 Moving만 (기본 전부 해제)
const checked = ref<Set<string>>(new Set())
function toggleCheck(id: string) {
  const s = new Set(checked.value)
  if (s.has(id)) s.delete(id)
  else s.add(id)
  checked.value = s
}
const canRegisterSel = computed(() => canRegisterAll.value && checked.value.size > 0)

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

async function clearAll() {
  if (!images.value.length) return
  if (!window.confirm('기준·Moving 이미지를 모두 비울까요? (마스크·정합 결과 포함)')) return
  await fetch('/api/reset', { method: 'POST' })
  selectedId.value = null
  checked.value = new Set()
  viewMode.value = 'mask'
  msg.value = '비웠습니다'
  await refresh()
}

async function startRegister(only: string[] | null = null) {
  msg.value = ''
  const res = await fetch('/api/register', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lazy: lazy.value, profile: profile.value, only }),
  })
  const d = await res.json()
  if (!res.ok) { msg.value = d.detail ?? '실행 실패'; return }
  running.value = true
}

function select(id: string, mode?: 'mask' | 'result') {
  selectedId.value = id
  // 탭 유지: 지정 없으면 보던 탭 그대로 (결과 없는 이미지면 마스크로)
  const want = mode ?? viewMode.value
  const img = images.value.find((i) => i.id === id)
  viewMode.value = want === 'result' && !img?.result ? 'mask' : want
  refresh() // 마스크 상태·미니맵 조건이 낡지 않게 전환 시 재조회
}

// 일괄 저장: 폴더 선택 → 결과 이미지 몰아서 저장 (전체/선택)
const anyResult = computed(() => images.value.some((i) => i.role === 'moving' && i.result))
const checkedWithResult = computed(() =>
  [...checked.value].filter((id) => images.value.find((i) => i.id === id)?.result))

async function saveResults(only: string[] | null) {
  msg.value = '저장 폴더 선택 창을 확인하세요…'
  try {
    const sel = await (await fetch('/api/select_folder', { method: 'POST' })).json()
    if (!sel.path) { msg.value = ''; return }
    const res = await fetch('/api/save_results', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dir: sel.path, only }),
    })
    const d = await res.json()
    if (!res.ok) throw new Error(d.detail ?? `HTTP ${res.status}`)
    msg.value = `${d.saved}장 저장됨 → ${d.dir}` +
      (d.failed.length ? ` (실패 ${d.failed.length})` : '')
  } catch (e: any) {
    msg.value = `저장 실패: ${e.message ?? e}`
  }
}

// GPU 가속 — 배포판 exe는 CPU torch 내장, GPU는 여기서 선택 설치
interface GpuInfo {
  device: string; gpu_name: string | null; installed: boolean; frozen: boolean
  installing: boolean; phase: string; done: number; total: number; error: string
}
const gpu = ref<GpuInfo | null>(null)

async function refreshGpu() {
  try { gpu.value = await (await fetch('/api/gpu')).json() } catch { /* 무시 */ }
}

const gpuPct = computed(() => {
  const g = gpu.value
  return g?.total ? Math.floor((g.done / g.total) * 100) : 0
})

async function installGpu() {
  if (!window.confirm(
    'GPU 가속을 설치할까요?\n\n' +
    '· 정합 속도 약 10배, 마스크 반응 약 4배 빨라집니다 (실측 RTX 4080)\n' +
    '· 약 2.5GB 다운로드 (최초 1회) · 설치 후 앱을 다시 시작하면 적용됩니다\n' +
    '· 설치 중에도 앱은 계속 사용할 수 있습니다')) return
  const res = await fetch('/api/gpu/install', { method: 'POST' })
  if (!res.ok) { msg.value = (await res.json()).detail ?? 'GPU 설치 실패'; return }
  await refreshGpu()
}

function badge(r: ImgInfo['result']): { text: string; cls: string } | null {
  if (!r) return null
  if (r.status === 'pass') return { text: 'PASS', cls: 'ok' }
  if (r.status === 'warn') return { text: 'WARN', cls: 'warn' }
  return { text: 'FAIL', cls: 'fail' }
}

onMounted(() => {
  refresh()
  refreshGpu()
  // 서버 재시작 등으로 낡아진 브라우저 상태 방지 (Register 409 예방)
  window.addEventListener('focus', refresh)
  es = new EventSource('/api/events')
  es.addEventListener('gpu', (e) => {
    const d = JSON.parse((e as MessageEvent).data)
    if (gpu.value) gpu.value = { ...gpu.value, ...d }
    if (d.phase === 'done' || d.phase === 'error') refreshGpu()
  })
  es.addEventListener('register', (e) => {
    const d = JSON.parse((e as MessageEvent).data)
    if (d.state === 'progress') msg.value = `정합 중 ${d.done + 1}/${d.total}: ${d.name}`
    else if (d.state === 'lazy') msg.value = `Lazy ${d.lazy_cur}/${d.lazy_total} (${d.lazy_label}) — ${d.done + 1}/${d.total}`
    else if (d.state === 'one_done') refresh()
    else if (d.state === 'done') {
      msg.value = `정합 완료 (${d.total}장)`
      running.value = false
      refresh().then(() => {
        // 성공한 항목은 체크 해제, 실패한 항목은 유지 (재시도 편의)
        checked.value = new Set([...checked.value].filter((id) =>
          images.value.find((i) => i.id === id)?.result?.status === 'fail'))
      })
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
        <button class="clear-btn" :disabled="!images.length" title="기준·Moving 모두 비우기"
                @click="clearAll">🗑</button>
      </div>

      <div class="img-list">
        <div
          v-for="img in images"
          :key="img.id"
          class="img-item"
          :class="{ active: img.id === selectedId, fixed: img.role === 'fixed' }"
          @click="select(img.id)"
        >
          <input
            v-if="img.role === 'moving'"
            type="checkbox"
            class="sel-chk"
            :checked="checked.has(img.id)"
            title="선택 정합 대상"
            @click.stop="toggleCheck(img.id)"
          />
          <img :src="`/api/image/${img.id}`" />
          <div class="img-meta">
            <div class="img-name">
              <span v-if="img.role === 'fixed'" class="role-tag">기준</span>{{ img.name }}
            </div>
            <div class="img-sub">
              <span :class="{ 'mask-ok': img.mask_ready }">
                {{ img.mask_ready ? '마스크 ✓' : '마스크 없음' }}
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
        <button class="register-btn" :disabled="!canRegisterAll" @click="startRegister(null)"
                title="모든 Moving을 기준에 정합 (마스크는 있는 페어만 사용)">
          {{ running ? '정합 중…'
             : images.some((i) => i.result) ? '▶ 전체 다시 정합' : '▶ 전체 정합' }}
        </button>
        <button class="register-btn sel" :disabled="!canRegisterSel"
                :title="checked.size ? '' : 'Moving 행의 체크박스를 선택하면 활성화됩니다'"
                @click="startRegister([...checked])">
          ▶ 선택 정합{{ checked.size ? ` (${checked.size})` : '' }}
        </button>
        <div class="save-row">
          <button class="save-btn" :disabled="running || !anyResult"
                  title="모든 정합 결과를 지정 폴더에 저장" @click="saveResults(null)">
            💾 전체 저장
          </button>
          <button class="save-btn" :disabled="running || !checkedWithResult.length"
                  title="체크된 이미지의 정합 결과만 저장"
                  @click="saveResults(checkedWithResult)">
            💾 선택 저장{{ checkedWithResult.length ? ` (${checkedWithResult.length})` : '' }}
          </button>
        </div>
        <div v-if="!running && images.length" class="statusmsg">
          마스크 없이도 정합됩니다 — 기준·Moving 양쪽에 마스크가 있으면 마스크 정합
        </div>
        <div v-if="gpu" class="gpu-row">
          <span v-if="gpu.device === 'cuda'" class="gpu-on">⚡ GPU 가속 사용 중</span>
          <span v-else-if="gpu.device === 'mps'" class="gpu-on">⚡ Metal(MPS) 가속 사용 중</span>
          <template v-else-if="gpu.installing">
            <span class="gpu-progress">
              GPU 가속 설치 중 {{ gpuPct }}%
              <template v-if="gpu.total"> ({{ (gpu.done / 1e9).toFixed(1) }}/{{ (gpu.total / 1e9).toFixed(1) }}GB)</template>
            </span>
          </template>
          <template v-else-if="gpu.installed">
            <span class="gpu-on">✓ 설치됨 — 앱을 다시 시작하면 적용</span>
          </template>
          <template v-else-if="gpu.gpu_name && gpu.frozen">
            <button class="gpu-btn" :title="`${gpu.gpu_name} 감지 — 약 2.5GB 다운로드`"
                    @click="installGpu">⚡ GPU 가속 켜기</button>
          </template>
          <span v-else-if="gpu.gpu_name" class="gpu-hint">GPU 감지됨 (소스 실행: CUDA torch 설치 시 자동 사용)</span>
          <span v-else class="gpu-hint">CPU 모드</span>
          <span v-if="gpu.error" class="gpu-err">설치 실패: {{ gpu.error }}</span>
        </div>
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
        <MaskEditor v-if="viewMode === 'mask'" :key="selected.id" :img="selected" :fixed="fixedImg"
                    @changed="refresh" @goto-fixed="fixedImg && select(fixedImg.id, 'mask')" />
        <ResultViewer v-else-if="selected.result" :key="'r' + selected.id" :img="selected" @changed="refresh" />
      </template>
      <p v-else class="hint center">← 이미지를 업로드하세요</p>
    </main>
  </div>
</template>
