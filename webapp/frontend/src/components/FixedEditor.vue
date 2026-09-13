<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import PhotoEditorCanvas from './PhotoEditorCanvas.vue'
import EditorControls from './EditorControls.vue'
import { getEditorPreview } from '../editorPreview'
import { freshEdits, renderEditedPNG, type Edits } from '../photoEdits'

const props = defineProps<{ image: { id: string; name: string; revision: number | string; source_w?: number; source_h?: number; edits?: Partial<Edits> | null }; busy?: boolean }>()
const emit = defineEmits<{ applied: []; cancel: []; error: [message: string] }>()
const root = ref<HTMLElement | null>(null)
const canvas = ref<InstanceType<typeof PhotoEditorCanvas> | null>(null)
const photo = ref<{ id: string; name: string; file: File; sourceWidth: number; sourceHeight: number } | null>(null)
const draft = ref<Edits>(freshEdits()), undoStack = ref<Edits[]>([]), redoStack = ref<Edits[]>([])
const previewBusy = ref(false)
const loading = ref(true), applying = ref(false), loadError = ref('')
const mode = ref<'view' | 'crop'>('view'), straighten = ref(false), before = ref(false), wholeOriginal = ref(false)
const cropRatio = ref<number | null>(null)
const copy = (e: Edits): Edits => JSON.parse(JSON.stringify(e))
const disabled = computed(() => loading.value || previewBusy.value || applying.value || props.busy)
let initial = freshEdits(), cropStart: Edits | null = null, gesture = false, recorded = false
let controller: AbortController | null = null
let baseRevision: number | string = props.image.revision
const dirty = computed(() => JSON.stringify(draft.value) !== JSON.stringify(initial))
const viewEdits = computed(() => wholeOriginal.value ? freshEdits() : draft.value)

async function load() {
  controller?.abort(); controller = new AbortController()
  const signal = controller.signal
  const target = {id: props.image.id, name: props.image.name, revision: props.image.revision, edits: {...props.image.edits}}
  loading.value = true; loadError.value = ''; photo.value = null
  try {
    const loaded = await getEditorPreview(target)
    if (signal.aborted) return
    photo.value = loaded
    initial = { ...freshEdits(), ...target.edits }
    baseRevision = target.revision
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
function controlChange(key:'brightness'|'contrast'|'rot90'|'flipH'|'flipV',value:number|boolean) {
  if(key==='flipH'||key==='flipV') set(key,Boolean(value)); else set(key,Number(value))
}
async function apply() {
  if (!photo.value || disabled.value || mode.value === 'crop') return
  const target = { id: props.image.id, revision: baseRevision, file: photo.value.file, edits: copy(draft.value) }
  applying.value = true
  try {
    const source = await fetch(`/api/image/${target.id}/source`, {signal:controller?.signal})
    if(!source.ok) throw new Error('적용할 원본 사진을 읽지 못했습니다.')
    const original = await source.blob()
    const rendered = await renderEditedPNG(new File([original],props.image.name,{type:original.type}), target.edits)
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
    <header class="fe-header"><div><span class="fe-eyebrow">PHOTO EDITOR 0.3 · 기준 사진 편집</span><strong>{{image.name}}</strong></div><span class="fe-note">{{applying?'원본 해상도로 적용 중…':'적용 전까지 정합 기준은 유지됩니다.'}}</span><button @click="cancel" :disabled="applying">편집 취소</button><button class="fe-primary" @click="apply" :disabled="disabled || mode==='crop'">기준에 적용</button></header>
    <div class="fe-body">
      <main>
        <PhotoEditorCanvas v-if="photo" ref="canvas" :key="photo.id" :photo="photo" :edits="viewEdits" :mode="wholeOriginal?'view':mode" :crop-ratio="cropRatio" :straighten="straighten && !wholeOriginal" :show-original="before" @angle="angle" @fine-deg="fine" @error="emit('error',$event)" @busy="previewBusy=$event"/>
        <div v-else class="fe-loading" role="status">{{loading?'편집 미리보기를 준비하는 중…':loadError}}<button v-if="loadError" @click="load">다시 시도</button></div>
        <span v-if="before || wholeOriginal" class="fe-comparison">{{wholeOriginal?'원본 전체 표시 중':'보정 전 톤 표시 중 · 구도 유지'}}</span>
        <div class="fe-view-tools"><button :disabled="!undoStack.length || disabled" @click="undo()">실행취소 · Ctrl/Cmd+Z</button><button :disabled="!redoStack.length || disabled" @click="undo(true)">다시 실행 · Ctrl/Cmd+Shift+Z</button><span class="fe-spacer"/><button :aria-pressed="before" @click="before=!before">보정 전 밝기·대비 (/)</button><button :aria-pressed="wholeOriginal" :disabled="mode==='crop'" @click="wholeOriginal=!wholeOriginal">원본 전체 보기</button><button @click="canvas?.fitView()">화면에 맞춤 (0)</button></div>
      </main>
      <aside><EditorControls :edits="draft" :mode="mode" :crop-ratio="cropRatio" :straighten="straighten" :disabled="disabled || wholeOriginal" @change="controlChange" @fine="fine" @crop="enterCrop" @apply="cropApply()" @cancel="cropCancel" @clear="cropApply(true)" @ratio="cropRatio=$event" @straighten="straighten=!straighten" @reset="reset" @gesture="$event?beginGesture():endGesture()"/></aside>
    </div>
  </section>
</template>
<style scoped>
.fixed-editor{display:flex;flex-direction:column;min-width:0;min-height:0;height:100%;background:#141d24;color:#e7edf2;outline:none}.fe-header{display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid #35424d;flex-wrap:wrap}.fe-header>div{display:flex;flex-direction:column;min-width:130px;max-width:320px}.fe-header strong{font-size:13px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.fe-eyebrow{font-size:10px;letter-spacing:.5px;color:#9ec9b6}.fe-note{flex:1;font-size:11px;color:#a1b1be}.fe-body{display:flex;flex:1;min-height:0}.fe-body main{display:flex;flex-direction:column;flex:1;min-width:0;position:relative;background:#0c1116}.fe-body main :deep(.editor-canvas){flex:1;height:auto;min-height:100px}aside{width:270px;flex-shrink:0;overflow:auto;padding:18px;border-left:1px solid #35424d;background:#1c252d}button{min-height:32px;padding:5px 9px;border:1px solid #465461;border-radius:6px;background:#28333e;color:inherit;font:inherit;font-size:11px;cursor:pointer}button:hover{border-color:#acd8c6}button:disabled{opacity:.4;cursor:default}button[aria-pressed=true]{border-color:#acd8c6;background:#2b493f}button:focus-visible{outline:2px solid #acd8c6;outline-offset:2px}.fe-primary{background:#acd8c6;border-color:#acd8c6;color:#142d23;font-weight:600}.fe-view-tools{display:flex;align-items:center;gap:5px;padding:7px 10px;border-top:1px solid #35424d;flex-shrink:0;flex-wrap:wrap}.fe-spacer{flex:1}.fe-loading{flex:1;display:flex;align-items:center;justify-content:center;gap:10px;color:#a9bac8;font-size:13px}.fe-comparison{position:absolute;top:12px;left:12px;padding:5px 8px;background:#203a30;border:1px solid #97c3b0;color:#c9e7d9;font-size:11px;border-radius:5px;pointer-events:none}@media(max-width:1100px){aside{width:230px;padding:12px}.fe-note{display:none}.fe-view-tools button{font-size:10px}.fe-header>div{flex:1}}@media(max-width:750px){aside{width:205px;padding:10px}.fe-header>div{max-width:190px}.fe-view-tools{gap:4px;padding:5px}}
</style>
