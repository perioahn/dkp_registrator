<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import PhotoEditorCanvas from './PhotoEditorCanvas.vue'
import { freshEdits, renderEditedPNG, type Edits } from '../photoEdits'

const props = defineProps<{ image: { id: string; name: string; revision: number | string; edits?: Partial<Edits> | null }; busy?: boolean }>()
const emit = defineEmits<{ applied: []; cancel: []; error: [message: string] }>()
const root = ref<HTMLElement | null>(null)
const canvas = ref<InstanceType<typeof PhotoEditorCanvas> | null>(null)
const photo = ref<{ id: string; name: string; file: File } | null>(null)
const draft = ref<Edits>(freshEdits()), undoStack = ref<Edits[]>([]), redoStack = ref<Edits[]>([])
const loading = ref(true), applying = ref(false), loadError = ref('')
const mode = ref<'view' | 'crop'>('view'), straighten = ref(false), before = ref(false), wholeOriginal = ref(false)
const cropRatio = ref<number | null>(null), customW = ref<number | null>(null), customH = ref<number | null>(null)
const ratios: [string, number][] = [['1:1',1],['4:3',4/3],['3:2',3/2],['16:10',1.6],['16:9',16/9]]
const copy = (e: Edits): Edits => JSON.parse(JSON.stringify(e))
const disabled = computed(() => loading.value || applying.value || props.busy)
let initial = freshEdits(), cropStart: Edits | null = null, gesture = false, recorded = false
let controller: AbortController | null = null
let baseRevision: number | string = props.image.revision
const dirty = computed(() => JSON.stringify(draft.value) !== JSON.stringify(initial))
const viewEdits = computed(() => wholeOriginal.value ? freshEdits() : draft.value)

async function load() {
  controller?.abort(); controller = new AbortController()
  const signal = controller.signal
  loading.value = true; loadError.value = ''; photo.value = null
  try {
    const response = await fetch(`/api/image/${props.image.id}/source`, { signal })
    if (!response.ok) throw new Error('원본 사진을 읽지 못했습니다.')
    const blob = await response.blob()
    if (signal.aborted) return
    photo.value = { id: props.image.id, name: props.image.name, file: new File([blob], props.image.name, { type: 'image/png' }) }
    initial = { ...freshEdits(), ...props.image.edits }
    baseRevision = props.image.revision
    draft.value = copy(initial); undoStack.value = []; redoStack.value = []
    mode.value = 'view'; before.value = false; wholeOriginal.value = false; cropStart = null
  } catch (e: any) { if (!signal.aborted) { loadError.value = e.message; emit('error', e.message) } }
  finally { if (!signal.aborted) loading.value = false }
}
function beginGesture() { gesture = true; recorded = false }
function endGesture() { gesture = false; recorded = false }
function record(value = draft.value) {
  if (mode.value === 'crop' || (gesture && recorded)) return
  undoStack.value.push(copy(value)); if (undoStack.value.length > 50) undoStack.value.shift()
  redoStack.value = []; recorded = gesture
}
function set<K extends keyof Edits>(key: K, value: Edits[K]) {
  if (disabled.value || wholeOriginal.value || JSON.stringify(draft.value[key]) === JSON.stringify(value)) return
  record(); draft.value[key] = value
}
function fine(value: number) {
  if (disabled.value || wholeOriginal.value) return
  const deg = Math.max(-15, Math.min(15, Math.round(value * 10) / 10))
  if (deg === draft.value.fineDeg) return
  record(); const crop = canvas.value?.cropForFineDeg(deg)
  draft.value.fineDeg = deg; if (crop) draft.value.crop = crop
}
function enterCrop() {
  if (disabled.value || wholeOriginal.value) return
  endGesture(); cropStart = copy(draft.value); mode.value = 'crop'; straighten.value = false
}
function cropApply(clear = false) {
  const crop = clear ? null : canvas.value?.currentCrop()
  if (crop === undefined) return
  mode.value = 'view'; if (cropStart) record(cropStart)
  draft.value.crop = crop; cropStart = null
}
function cropCancel() {
  if (cropStart) draft.value = copy(cropStart)
  cropStart = null; mode.value = 'view'; canvas.value?.syncRectFromEdits()
}
function undo(redo = false) {
  if (disabled.value) return
  if (mode.value === 'crop') { cropCancel(); return }
  endGesture()
  const from = redo ? redoStack : undoStack, to = redo ? undoStack : redoStack
  const item = from.value.pop()
  if (item) { to.value.push(copy(draft.value)); draft.value = item }
}
function reset() { if (!disabled.value) { record(); draft.value = freshEdits() } }
function angle(deg: number) { fine(draft.value.fineDeg - deg); straighten.value = false }
function customRatio() { if (customW.value && customH.value && customW.value > 0 && customH.value > 0) cropRatio.value = customW.value / customH.value }
function numeric(key: 'brightness'|'contrast'|'fineDeg', event: Event) {
  const value = (event.target as HTMLInputElement).valueAsNumber
  if (!Number.isFinite(value)) return
  if (key==='fineDeg') fine(value)
  else set(key,Math.max(-100,Math.min(100,value)))
}
async function apply() {
  if (!photo.value || disabled.value || mode.value === 'crop') return
  const target = { id: props.image.id, revision: baseRevision, file: photo.value.file, edits: copy(draft.value) }
  applying.value = true
  try {
    const rendered = await renderEditedPNG(target.file, target.edits)
    const data = new FormData()
    data.append('image', rendered.blob, 'edited.png')
    data.append('metadata', JSON.stringify({ edits: rendered.edits, G: rendered.G, width: rendered.width,
      height: rendered.height, base_revision: target.revision }))
    const response = await fetch(`/api/image/${target.id}/edit`, { method: 'POST', body: data })
    if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || '편집 적용 실패') }
    emit('applied')
  } catch (e: any) { emit('error', e.message ?? String(e)) }
  finally { applying.value = false }
}
function cancel() { if (!applying.value) emit('cancel') }
function key(e: KeyboardEvent) {
  // Stop bubbling even when an input owns its keys: the workspace must not undo masks.
  e.stopPropagation()
  if (e.isComposing || e.repeat || (e.target as HTMLElement).closest('input,textarea,select,[contenteditable="true"]')) return
  if (disabled.value) return
  if (e.ctrlKey || e.metaKey) {
    if (e.code === 'KeyZ') { e.preventDefault(); undo(e.shiftKey) }
    else if (e.code === 'KeyS') { e.preventDefault(); apply() }
    return
  }
  if (e.altKey) return
  if (e.code === 'Escape') { e.preventDefault(); if (mode.value === 'crop') cropCancel(); else straighten.value = false }
  else if (e.code === 'Enter' && mode.value === 'crop') { e.preventDefault(); cropApply() }
  else if (e.code === 'KeyR') { e.preventDefault(); mode.value === 'crop' ? cropCancel() : enterCrop() }
  else if (e.code === 'Slash') { e.preventDefault(); before.value = !before.value }
  else if (e.code === 'Digit0') { e.preventDefault(); canvas.value?.fitView() }
}
function unload(e: BeforeUnloadEvent) { if (dirty.value) { e.preventDefault(); e.returnValue = '' } }
watch(() => props.image.id, load)
onMounted(() => { load(); root.value?.focus({ preventScroll: true }); window.addEventListener('beforeunload', unload) })
onUnmounted(() => { controller?.abort(); window.removeEventListener('beforeunload', unload) })
defineExpose({ undo, apply, dirty })
</script>

<template>
  <section ref="root" class="fixed-editor" tabindex="-1" data-tool="fixed-editor" @keydown="key" :aria-busy="disabled">
    <header class="fe-header">
      <div><strong>기준 사진 편집 중</strong><span>{{ image.name }}</span></div>
      <span class="fe-note">{{ applying ? '원본 해상도로 적용 중…' : '원본을 보존하며, 적용 전까지 정합 기준은 유지됩니다.' }}</span>
      <button @click="cancel" :disabled="applying">편집 취소</button>
      <button class="fe-primary" @click="apply" :disabled="disabled || mode === 'crop'">기준에 적용</button>
    </header>
    <div class="fe-tools">
      <template v-if="mode === 'crop'">
        <strong>크롭 영역</strong><button :aria-pressed="cropRatio === null" @click="cropRatio = null">자유</button>
        <button v-for="[label,ratio] in ratios" :key="label" :aria-pressed="cropRatio === ratio" @click="cropRatio = ratio">{{ label }}</button>
        <label class="fe-ratio"><input v-model.number="customW" type="number" min="1" aria-label="크롭 가로 비율" placeholder="가로" @change="customRatio">:<input v-model.number="customH" type="number" min="1" aria-label="크롭 세로 비율" placeholder="세로" @change="customRatio"></label>
        <button @click="cropApply(true)">크롭 제거</button><button @click="cropCancel">영역 취소</button><button class="fe-primary" @click="cropApply()">영역 적용</button>
      </template>
      <template v-else>
        <button :disabled="disabled || wholeOriginal" @click="set('rot90',(draft.rot90+3)%4)">↶ 90°</button>
        <button :disabled="disabled || wholeOriginal" @click="set('rot90',(draft.rot90+1)%4)">↷ 90°</button>
        <button :disabled="disabled || wholeOriginal" @click="set('flipH',!draft.flipH)">좌우 반전</button>
        <button :disabled="disabled || wholeOriginal" @click="set('flipV',!draft.flipV)">상하 반전</button>
        <button :disabled="disabled || wholeOriginal" :aria-pressed="straighten" @click="straighten=!straighten">수평선 긋기</button>
        <button :disabled="disabled || wholeOriginal" @click="enterCrop">크롭 (R)</button>
      </template>
      <button @click="canvas?.fitView()">화면에 맞춤 (0)</button>
    </div>
    <div class="fe-body">
      <main>
        <PhotoEditorCanvas v-if="photo" ref="canvas" :key="photo.id" :photo="photo" :edits="viewEdits" :mode="wholeOriginal ? 'view' : mode" :crop-ratio="cropRatio" :straighten="straighten && !wholeOriginal" :show-original="before" @angle="angle" @fine-deg="fine" @error="emit('error',$event)" />
        <div v-else class="fe-loading" role="status">{{ loading ? '원본 사진을 불러오는 중…' : loadError }}<button v-if="loadError" @click="load">다시 시도</button></div>
      </main>
      <aside>
        <h3>밝기와 구도</h3>
        <label v-for="[label,key,min,max,step] in [['밝기','brightness',-100,100,1],['대비','contrast',-100,100,1],['미세 회전','fineDeg',-15,15,0.1]] as const" :key="key">
          <span>{{ label }} <output>{{ draft[key] }}{{ key==='fineDeg' ? '°' : '' }}</output></span>
          <input type="range" :aria-label="label" :value="draft[key]" :min="min" :max="max" :step="step" :disabled="disabled || wholeOriginal" @pointerdown="beginGesture" @pointerup="endGesture" @change="endGesture" @pointercancel="endGesture" @blur="endGesture" @input="key==='fineDeg' ? fine(+($event.target as HTMLInputElement).value) : set(key,+($event.target as HTMLInputElement).value)">
          <button class="fe-small" @click="key==='fineDeg' ? fine(0) : set(key,0)">기본값</button>
          <input type="number" :aria-label="`${label} 수치`" :value="draft[key]" :min="min" :max="max" :step="step" :disabled="disabled || wholeOriginal" @focus="beginGesture" @blur="endGesture" @input="numeric(key,$event)">
        </label>
        <button :aria-pressed="before" @click="before=!before">보정 전 밝기·대비 (/)</button>
        <button :aria-pressed="wholeOriginal" :disabled="mode==='crop'" @click="wholeOriginal=!wholeOriginal">원본 전체 보기</button>
        <p v-if="before || wholeOriginal" role="status">{{ wholeOriginal ? '원본 전체를 보는 중입니다.' : '구도는 유지하고 보정 전 톤을 보는 중입니다.' }}</p>
        <button :disabled="!undoStack.length || disabled" @click="undo()">실행취소 · Ctrl/Cmd+Z</button>
        <button :disabled="!redoStack.length || disabled" @click="undo(true)">다시 실행 · Ctrl/Cmd+Shift+Z</button>
        <button :disabled="disabled" @click="reset">모든 보정 초기화</button>
        <p>휠: 포인터 중심 확대<br>Space+드래그: 화면 이동<br>크롭은 영역만 자릅니다.</p>
      </aside>
    </div>
  </section>
</template>

<style scoped>
.fixed-editor { display:flex;flex-direction:column;min-width:0;min-height:0;height:100%;background:#15191f;color:#e8edf3;outline:none }
.fe-header,.fe-tools { display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:10px 12px;border-bottom:1px solid #343b45 }
.fe-header>div { display:flex;flex-direction:column;gap:3px;min-width:130px }
.fe-header span,.fe-note { font-size:12px;color:#b0bac7 }
.fe-note { flex:1 }.fe-header>div>span { max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap }
button { min-height:36px;padding:6px 10px;border:1px solid #465262;border-radius:6px;background:#252c36;color:inherit;cursor:pointer;font:inherit;font-size:12px }
button:hover { border-color:#8fbdf0 }button:disabled { opacity:.45;cursor:default }button:focus-visible,input:focus-visible { outline:2px solid #82b7ef;outline-offset:2px }
button[aria-pressed=true] { border-color:#7db6ee;background:#28445e }.fe-primary { background:#28669c;border-color:#438cc8 }
.fe-body { display:flex;flex:1;min-height:0 }.fe-body main { flex:1;min-width:0;background:#0c1015 }
aside { width:230px;flex-shrink:0;padding:14px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;border-left:1px solid #343b45 }
h3 { font-size:13px;margin:0 0 4px }label { display:block;font-size:12px }label>span { display:flex;justify-content:space-between }input[type=range] { width:100%;accent-color:#7fb0df;margin:10px 0 0 }
.fe-small { min-height:26px;padding:2px 7px;font-size:11px;float:right }.fe-ratio { display:flex;gap:3px;align-items:center }.fe-ratio input { width:55px;min-height:32px;background:#161d26;color:inherit;border:1px solid #465262;border-radius:4px;padding:5px }
aside p { font-size:11px;line-height:1.7;color:#b0bac7;margin:0 }.fe-loading { display:flex;gap:10px;align-items:center;justify-content:center;height:100% }
@media(max-width:1050px) { aside { width:190px;padding:10px }.fe-note { display:none } }
@media(max-width:700px) { .fe-body { flex-direction:column }.fe-body main { min-height:200px }aside { width:auto;max-height:190px;border-left:0;border-top:1px solid #343b45;display:grid;grid-template-columns:repeat(2,minmax(0,1fr)) }.fe-header>div>span { max-width:150px } }
</style>
