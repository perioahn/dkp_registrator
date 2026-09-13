<script setup lang="ts">
import { computed, nextTick, onUnmounted, ref, watch } from "vue";
import PhotoViewport from "./PhotoViewport.vue";
import {
  api,
  mapPoint,
  inversePoint,
  type Photo,
  type Tool,
  type ComparisonMode,
  type Anchor,
  type Point,
  type Viewport,
} from "../workspace";
import { views, viewFor, adjustments } from "../viewstate";
const props = defineProps<{
  fixed: Photo;
  current: Photo;
  tool: Tool;
  mode: ComparisonMode;
  running: boolean;
  revision: number;
  registrationError?: string;
  recoveryAvailable: boolean;
}>();
const emit = defineEmits<{
  changed: [];
  error: [string];
  "update:tool": [Tool];
  recover: [{mid: string; fixedId: string; revision: number}];
  "recovery-busy": [boolean];
}>();
const leftPane = ref<InstanceType<typeof PhotoViewport>>(),
  rightPane = ref<InstanceType<typeof PhotoViewport>>();
const linked = ref(true),
  maskVisible = ref(true),
  opacity = ref(0.65),
  wipe = ref(50),
  busy = ref(false),
  maskTarget = ref("");
const pairs = ref<Anchor[]>([]),
  pairRevision = ref(0),
  selectedPair = ref<string | null>(null),
  placing = ref(false),
  pending = ref<Point | null>(null),
  showAnchors = ref(true),
  anchorsExpanded = ref(false);
// Tool preference survives navigation; coordinates and saved pairs do not.
const manualAnchorMode = ref(false);
const recommendation = ref<{token: string; missing: string[]} | null>(null);
const recommending = ref(false), guidance = ref(''), guidanceError = ref('');
const recommendationOwner = ref('');
const anchorControls = ref<HTMLElement>();
const comparisonArea = ref<HTMLElement>();
const inputKey = computed(() => [props.fixed.id, props.fixed.revision, props.fixed.mask_rev,
  props.current.id, props.current.revision, props.current.mask_rev].join(':'));
let recommendationSequence = 0;
const recoveryBlocked = computed(() => props.running || busy.value || recommending.value);
const hasRecoveryMask = computed(() => !!r.value?.used_mask || props.fixed.mask_ready || props.current.mask_ready);
const defaultRecovery = computed(() => props.fixed.mask_ready && props.current.mask_ready ? 'recommend' : 'mask');
const enabledPairs = computed(() => pairs.value.filter(p => p.enabled).length);
const guidanceText = computed(() => {
  if (!props.recoveryAvailable) return '실행 중인 서버가 이전 버전입니다. 자동 앵커 기능을 사용하려면 실행 바로가기를 다시 열어 새 버전으로 접속하세요. 현재 작업은 이 창에 유지됩니다.';
  if (recommending.value) return recommendationOwner.value === inputKey.value
    ? '합쳐진 마스크 영역에서 대응점을 고르게 찾고 있습니다.' : '다른 사진의 앵커 추천을 마무리하고 있습니다.';
  if (props.running) return '정합을 계산하고 있습니다. 완료되면 여기서 결과를 확인하세요.';
  if (guidanceError.value || props.registrationError) return guidanceError.value || props.registrationError;
  if (guidance.value) return guidance.value;
  if (manualAnchorMode.value && props.current.id !== props.fixed.id)
    return '기준 사진의 한 점을 클릭하고 A를 누른 뒤, 현재 사진의 같은 위치를 클릭하고 A를 누르세요.';
  if (r.value?.latest_attempt_failed && hasResult.value)
    return '이번 정합은 실패했습니다. 이전 결과를 유지하고 있습니다. 아래 방법으로 보정해 보세요.';
  if (r.value?.status === 'fail') return '정합 결과를 만들지 못했습니다. 앵커를 지정하거나 마스크를 다시 선택해 보세요.';
  if (r.value?.reference_conflict) return '선택한 앵커들이 서로 잘 맞지 않습니다. 점의 짝을 확인하거나 마스크를 다시 선택해 보세요.';
  return '정합 결과가 마음에 들지 않으면 아래 방법으로 보정해 보세요.';
});
const guidanceDetail = computed(() => r.value?.latest_attempt_reason || r.value?.reason || '');
const missingRegions = computed(() => {
  const missing = recommendation.value?.missing || [];
  return (['fixed', 'moving'] as const).filter(side => missing.some(item => item.startsWith(side+':')))
    .map(side => `${side === 'fixed' ? '기준 사진' : '현재 사진'}의 일부 마스크 영역`).join(' · ');
});
watch(inputKey, () => {
  recommendationSequence++;
  recommendation.value = null;
  guidance.value = '';
  guidanceError.value = '';
  pairs.value = [];
  void loadAnchors();
});
watch(() => props.current.result?.id, () => { guidance.value = ''; guidanceError.value = ''; });
watch(() => props.tool, t => {
  if (t !== 'mask' && t !== 'anchor') manualAnchorMode.value = false;
  if (t === 'compare') guidance.value = '';
});
async function openManualAnchors() {
  if (recoveryBlocked.value) return;
  const key = inputKey.value;
  manualAnchorMode.value = true;
  emit('update:tool', 'mask');
  anchorsExpanded.value = true;
  showAnchors.value = true;
  guidanceError.value = '';
  guidance.value = '기준 사진의 한 점을 클릭하고 A를 누른 뒤, 현재 사진의 같은 위치를 클릭하고 A를 누르세요.';
  await loadAnchors();
  if (key !== inputKey.value) return;
  await startAnchor();
  await nextTick();
  anchorControls.value?.focus({preventScroll: true});
}
async function suggestAnchors() {
  if (!props.recoveryAvailable || recoveryBlocked.value || props.current.id === props.fixed.id) return;
  const key = inputKey.value, mid = props.current.id, fid = props.fixed.id;
  const sequence = ++recommendationSequence;
  recommendationOwner.value = key;
  recommending.value = true;
  emit('recovery-busy', true);
  guidanceError.value = '';
  clearPreview();
  placing.value = false;
  pending.value = null;
  try {
    // Do not discard the user's current points or draft until a fresh response arrives.
    const saved = await api(`/api/anchors/${mid}`);
    const data = await api(`/api/anchors/${mid}/recommend`, {fixed_id: fid});
    if (disposed || sequence !== recommendationSequence || key !== inputKey.value) return;
    if (data.fixed_id !== fid || saved.revision !== data.revision) throw Error('앵커가 바뀌었습니다. 다시 추천받으세요.');
    manualAnchorMode.value = false;
    emit('update:tool', 'mask');
    anchorsExpanded.value = true;
    showAnchors.value = true;
    pairs.value = [...saved.pairs.filter((p:Anchor) => p.source !== 'automatic'), ...data.pairs];
    pairRevision.value = data.revision;
    recommendation.value = {token: data.token, missing: data.missing};
    selectedPair.value = null;
    guidance.value = data.pairs.length
      ? `추천 앵커 ${data.pairs.length}쌍을 표시했습니다. 같은 위치인지 확인한 뒤 ‘이 점들로 재정합’을 누르세요.`
      : '신뢰할 대응점을 찾지 못했습니다. 앵커를 직접 찍거나 마스크를 다시 선택해 주세요.';
    if (data.missing.length && data.pairs.length) guidance.value += ' 일부 마스크 영역은 추천점이 부족합니다.';
    await nextTick();
    anchorControls.value?.focus({preventScroll: true});
    comparisonArea.value?.scrollIntoView({block: 'nearest'});
  } catch (e:any) {
    if (!disposed && sequence === recommendationSequence && key === inputKey.value) guidanceError.value = e.message;
  } finally {
    recommending.value = false;
    emit('recovery-busy', false);
  }
}
async function discardRecommendation() {
  if (recoveryBlocked.value) return;
  recommendation.value = null;
  guidance.value = '';
  guidanceError.value = '';
  await loadAnchors();
}
async function recoverWithAnchors() {
  if (!props.recoveryAvailable || recoveryBlocked.value || enabledPairs.value < 2) return;
  const mid = props.current.id, fixedId = props.fixed.id, key = inputKey.value;
  if (recommendation.value && !await savePairs(pairs.value, true)) return;
  if (key !== inputKey.value) return;
  guidance.value = '';
  emit('recover', {mid, fixedId, revision: pairRevision.value});
}
function openRecoveryTool(tool: 'mask' | 'adjust') {
  if (recoveryBlocked.value) return;
  manualAnchorMode.value = false;
  pending.value = null;
  placing.value = false;
  guidanceError.value = '';
  guidance.value = tool === 'mask'
    ? '기준으로 삼을 구조물을 선택하고 Z로 확정한 뒤 다시 정합하세요.'
    : '위치·회전·균일 배율을 조절해 맞춰 보세요.';
  emit('update:tool', tool);
}
async function restorePrevious() {
  if (recoveryBlocked.value || !r.value?.has_previous) return;
  const key = pairKey.value;
  busy.value = true;
  try {
    await api(`/api/result/${props.current.id}/restore-previous`, {result_id: r.value.id});
    if (key === pairKey.value) { previous.value = false; guidance.value = '이전 정합 결과로 돌아왔습니다.'; }
    emit('changed');
  } catch (e:any) {
    if (key === pairKey.value) guidanceError.value = e.message;
  } finally { busy.value = false; }
}
const candidate = ref<{which: "left" | "right"; x: number; y: number} | null>(null);
const preview = ref<{token: string; overlay: string; imageId: string} | null>(null);
const previewBusy = ref(false);
let previewSequence = 0;
let previewPoints: {x:number; y:number; label:number}[] = [];
let previewImage = "";
let previewInFlight = false;
let queuedPreview: {imageId:string; points:typeof previewPoints; sequence:number} | null = null;
let disposed = false;
async function drainPreview() {
  if (previewInFlight || !queuedPreview || disposed) return;
  const job = queuedPreview;
  queuedPreview = null;
  previewInFlight = true;
  try {
    const result = await api(`/api/mask/${job.imageId}/preview`, {points:job.points});
    if (!disposed && job.sequence === previewSequence)
      preview.value = {...result, imageId:job.imageId};
  } catch (e:any) {
    if (!disposed && job.sequence === previewSequence) emit('error', e.message);
  } finally {
    previewInFlight = false;
    if (queuedPreview) void drainPreview();
    else if (job.sequence === previewSequence) previewBusy.value = false;
  }
}
function clearPreview() {
  previewSequence++;
  candidate.value = null;
  preview.value = null;
  previewBusy.value = false;
  previewPoints = [];
  previewImage = "";
  queuedPreview = null;
}
onUnmounted(() => { disposed = true; clearPreview(); });
function toggleAnchors() {
  anchorsExpanded.value = !anchorsExpanded.value;
  if (anchorsExpanded.value) void loadAnchors();
  else { manualAnchorMode.value = false; pending.value = null; placing.value = false; selectedPair.value = null; }
}
function maskOverlay(photo: Photo) {
  if (!isRaw.value || !maskVisible.value) return undefined;
  if (preview.value?.imageId === photo.id) return preview.value.overlay;
  if (!photo.mask_ready) return undefined;
  return `/api/mask/${photo.id}/overlay?v=${photo.revision}&mask=${photo.mask_rev}`;
}
function cancelInput() {
  if (!candidate.value && !preview.value && !previewBusy.value && !pending.value) return false;
  cancel();
  return true;
}
let draggingPairs: Anchor[] | null = null;
const pairKey = computed(() => `${props.fixed.id}:${props.current.id}`);
const r = computed(() => props.current.result),
  hasResult = computed(() => !!r.value && r.value.status !== "fail");
const isRaw = computed(
  () =>
    props.tool === "anchor" ||
    props.tool === "mask" ||
    !hasResult.value,
);
const resultKey = computed(
  () => `${r.value?.fixed_id ?? props.fixed.id}:${props.current.id}:${r.value?.id ?? ""}`,
);
const previous = ref(false);
watch(resultKey, () => (previous.value = false));
watch(
  () => props.tool,
  (t) => {
    if (t === "adjust") previous.value = false;
  },
);
const shownResult = computed(() =>
  previous.value && r.value?.previous ? r.value.previous : r.value,
);
const resultReference = computed(() => shownResult.value?.fixed_name || props.fixed.name);
const leftKey = computed(() =>
  isRaw.value
    ? `raw:${props.fixed.id}`
    : `result:${shownResult.value?.fixed_id}:${shownResult.value?.fixed_revision}`,
);
const rightKey = computed(() =>
  isRaw.value
    ? `raw:${props.current.id}`
    : leftKey.value + (linked.value ? "" : ":right"),
);
const leftView = computed(() => viewFor(leftKey.value)),
  rightView = computed(() => viewFor(rightKey.value));
const width = computed(() =>
  isRaw.value
    ? props.fixed.full_w
    : (shownResult.value?.full_w ?? props.fixed.full_w),
);
const height = computed(() =>
  isRaw.value
    ? props.fixed.full_h
    : (shownResult.value?.full_h ?? props.fixed.full_h),
);
const rightWidth = computed(() =>
    isRaw.value ? props.current.full_w : width.value,
  ),
  rightHeight = computed(() =>
    isRaw.value ? props.current.full_h : height.value,
  );
const version = computed(
  () => `result=${shownResult.value?.id ?? ""}`,
);
const base = computed(
  () => `/api/result/${props.current.id}${previous.value ? "/previous" : ""}`,
);
const matchSize = ref({ w: 1, h: 1 });
watch([base, version, () => props.mode], () => {
  if (props.mode !== "match") return;
  const url = `${base.value}/match_viz?${version.value}`,
    im = new Image();
  im.onload = () => {
    if (url === `${base.value}/match_viz?${version.value}`)
      matchSize.value = { w: im.naturalWidth, h: im.naturalHeight };
  };
  im.src = url;
});
const leftSrc = computed(() =>
  isRaw.value
    ? `/api/image/${props.fixed.id}?v=${props.fixed.revision}`
    : `${base.value}/fixed?${version.value}`,
);
const rightSrc = computed(() =>
  isRaw.value
    ? `/api/image/${props.current.id}?v=${props.current.revision}`
    : `${base.value}/registered?${version.value}`,
);
const leftRegion = computed(() =>
  isRaw.value
    ? `/api/image/${props.fixed.id}/region?v=${props.fixed.revision}`
    : `${base.value}/region?kind=fixed&${version.value}`,
);
const rightRegion = computed(() =>
  isRaw.value
    ? `/api/image/${props.current.id}/region?v=${props.current.revision}`
    : `${base.value}/region?kind=registered&${version.value}`,
);
const side = computed(
  () => isRaw.value || (props.mode === "side" && props.tool !== "adjust"),
);
const flick = ref(true);
const interval = setInterval(() => {
  if (props.mode === "flicker") flick.value = !flick.value;
}, 600);
onUnmounted(() => clearInterval(interval));
const adj = computed(
  () =>
    adjustments[resultKey.value] ??
    (adjustments[resultKey.value] = { dx: 0, dy: 0, rot: 0, scale: 1 }),
);
const adjDirty = computed(
  () =>
    adj.value.dx !== 0 ||
    adj.value.dy !== 0 ||
    adj.value.rot !== 0 ||
    adj.value.scale !== 1,
);
const adjFields = computed(() => [
  {
    key: "dx" as const,
    title: "좌우",
    min: -width.value / 2,
    max: width.value / 2,
    step: 1,
  },
  {
    key: "dy" as const,
    title: "상하",
    min: -height.value / 2,
    max: height.value / 2,
    step: 1,
  },
  { key: "rot" as const, title: "회전", min: -30, max: 30, step: 0.1 },
  {
    key: "scale" as const,
    title: "균일 배율",
    min: 0.5,
    max: 1.5,
    step: 0.001,
  },
]);
const adjHistory: Record<string, Array<typeof adj.value>> = {},
  adjRedo: Record<string, Array<typeof adj.value>> = {};
let gesture: typeof adj.value | null = null;
function startGesture() {
  if (!gesture) gesture = { ...adj.value };
}
function finishGesture() {
  if (gesture && JSON.stringify(gesture) !== JSON.stringify(adj.value)) {
    (adjHistory[resultKey.value] ??= []).push(gesture);
    adjRedo[resultKey.value] = [];
  }
  gesture = null;
}
function pointerAdjust(
  value: typeof adj.value,
  stage: "start" | "move" | "end" | "cancel",
) {
  if (busy.value || props.running) return;
  if (stage === "start") startGesture();
  Object.assign(adj.value, value);
  if (stage === "end") finishGesture();
  if (stage === "cancel") gesture = null;
}
function draftUndo(redo = false) {
  const from = (redo ? adjRedo : adjHistory)[resultKey.value] ?? [],
    to = ((redo ? adjHistory : adjRedo)[resultKey.value] ??= []);
  const last = from.pop();
  if (last) {
    to.push({ ...adj.value });
    adjustments[resultKey.value] = last;
    return true;
  }
  return false;
}
function updateView(which: "left" | "right", v: Viewport) {
  const key = which === "left" ? leftKey.value : rightKey.value;
  views[key] = v;
  if (linked.value && isRaw.value) {
    const other = which === "left" ? rightKey.value : leftKey.value;
    views[other] = { ...viewFor(other), zoom: v.zoom };
  }
}
function fit() {
  leftPane.value?.fit();
  rightPane.value?.fit();
}
let anchorRequest = 0;
async function loadAnchors() {
  const token = ++anchorRequest;
  if (!anchorsExpanded.value) return;
  if (recommendation.value) return;
  if (props.current.id === props.fixed.id) {
    pairs.value = [];
    return;
  }
  try {
    const d = await api(`/api/anchors/${props.current.id}`);
    if (token === anchorRequest) {
      pairs.value = d.pairs;
      pairRevision.value = d.revision;
    }
  } catch (e: any) {
    if (token === anchorRequest) emit("error", e.message);
  }
}
watch(
  [() => props.current.id, () => props.fixed.id],
  () => {
    clearPreview();
    placing.value = manualAnchorMode.value && (props.tool === 'mask' || props.tool === 'anchor')
      && props.current.id !== props.fixed.id;
    pending.value = null;
    selectedPair.value = null;
    pairs.value = [];
    draggingPairs = null;
    gesture = null;
    maskTarget.value = props.current.id;
    loadAnchors();
  },
  { immediate: true },
);
watch(
  () => props.revision,
  () => {
    clearPreview();
    if (!busy.value) loadAnchors();
  },
);
async function startAnchor() {
  if (!anchorsExpanded.value) return;
  if (props.current.id === props.fixed.id) return;
  manualAnchorMode.value = true;
  emit("update:tool", "mask");
  const pick = candidate.value;
  if (pick) {
    const img = pick.which === "left" ? props.fixed : props.current;
    const point = inversePoint(img.G, [pick.x, pick.y]);
    clearPreview();
    if (pick.which === "left") {
      pending.value = point;
      placing.value = true;
      selectedPair.value = null;
    } else if (pending.value) {
      const a = {id: crypto.randomUUID(), fixed: pending.value, moving: point, enabled: true};
      if (await savePairs([...pairs.value, a])) {
        selectedPair.value = a.id;
        pending.value = null;
        placing.value = false;
      }
    } else emit("error", "먼저 기준 사진을 클릭하고 A로 기준점을 선택하세요.");
    return;
  }
  placing.value = true;
  selectedPair.value = null;
}
async function savePairs(next: Anchor[], commit = false) {
  if (busy.value || props.running) return false;
  if (recommendation.value && !commit) { pairs.value = next; return true; }
  busy.value = true;
  emit('recovery-busy', true);
  const key = pairKey.value,
    mid = props.current.id,
    fid = props.fixed.id,
    rev = pairRevision.value;
  try {
    const data = await api(
      `/api/anchors/${mid}`,
      { pairs: next, base_revision: rev, fixed_id: fid, input_token: recommendation.value?.token },
      "PUT",
    );
    if (key === pairKey.value) {
      recommendation.value = null;
      pairs.value = data.pairs;
      pairRevision.value = data.revision;
    }
    emit("changed");
    return key === pairKey.value;
  } catch (e: any) {
    if (key === pairKey.value) guidanceError.value = e.message;
    if (key === pairKey.value) await loadAnchors();
    return false;
  } finally {
    busy.value = false;
    emit('recovery-busy', false);
  }
}
async function deleteAnchor() {
  if (busy.value || props.running) return;
  if (candidate.value) { clearPreview(); return; }
  if (pending.value || placing.value) {
    pending.value = null;
    placing.value = false;
    return;
  }
  if (selectedPair.value) {
    await savePairs(pairs.value.filter((p) => p.id !== selectedPair.value));
    selectedPair.value = null;
  }
}
function cancel() {
  clearPreview();
  pending.value = null;
  placing.value = false;
  if (gesture) {
    Object.assign(adj.value, gesture);
    gesture = null;
  }
}
async function click(
  which: "left" | "right",
  p: { x: number; y: number; button: number },
) {
  if (busy.value || props.running) return;
  const img = which === "left" ? props.fixed : props.current;
  if (isRaw.value && props.tool !== "adjust") {
    maskTarget.value = img.id;
    if (p.button !== 2) candidate.value = {which, x:p.x, y:p.y};
    if (previewImage !== img.id) { previewPoints = []; preview.value = null; }
    previewImage = img.id;
    if (previewPoints.length >= 100) { emit('error', '개체를 Z로 확정한 뒤 다음 영역을 선택하세요.'); return; }
    previewPoints.push({
      x: Math.max(0, Math.min(img.w - 1, ((p.x + .5) * img.w) / img.full_w - .5)),
      y: Math.max(0, Math.min(img.h - 1, ((p.y + .5) * img.h) / img.full_h - .5)),
      label: p.button === 2 ? 0 : 1,
    });
    const sequence = ++previewSequence;
    previewBusy.value = true;
    queuedPreview = {imageId:img.id, points:[...previewPoints], sequence};
    void drainPreview();
    return;
  }
  if (props.tool !== "anchor" || !placing.value || p.button !== 0) return;
  if (which === "left" && !pending.value)
    pending.value = inversePoint(img.G, [p.x, p.y]);
  else if (which === "right" && pending.value) {
    const a = {
      id: crypto.randomUUID(),
      fixed: pending.value,
      moving: inversePoint(img.G, [p.x, p.y]),
      enabled: true,
    };
    if (await savePairs([...pairs.value, a])) {
      selectedPair.value = a.id;
      placing.value = false;
      pending.value = null;
    }
  }
}
function points(which: "left" | "right") {
  if (!isRaw.value || !anchorsExpanded.value || !showAnchors.value) return [];
  const img = which === "left" ? props.fixed : props.current;
  const list = pairs.value
    .map((a, i) => {
      const p = mapPoint(img.G, which === "left" ? a.fixed : a.moving);
      return {
        id: a.id,
        x: p[0],
        y: p[1],
        label: String(i + 1),
        selected: a.id === selectedPair.value,
        disabled: !a.enabled,
      };
    })
    .filter(
      (p) => p.x >= 0 && p.y >= 0 && p.x <= img.full_w && p.y <= img.full_h,
    );
  if (which === "left" && pending.value) {
    const p = mapPoint(img.G, pending.value);
    list.push({
      id: "pending",
      x: p[0],
      y: p[1],
      label: "…",
      selected: true,
      disabled: false,
    });
  }
  if (candidate.value?.which === which) list.push({
    id: "candidate", x:candidate.value.x, y:candidate.value.y,
    label: "?", selected: true, disabled: false,
  });
  return list;
}
async function dragAnchor(
  which: "left" | "right",
  p: { id: string; x: number; y: number; end: boolean; moved?: boolean },
) {
  if (props.running || busy.value || p.id === "pending" || p.id === "candidate") return;
  selectedPair.value = p.id;
  if (p.end && !p.moved) return;
  if (!draggingPairs) draggingPairs = JSON.parse(JSON.stringify(pairs.value));
  const a = pairs.value.find((a) => a.id === p.id);
  if (!a) return;
  const img = which === "left" ? props.fixed : props.current,
    point = inversePoint(img.G, [
      Math.max(0, Math.min(img.full_w - 1, p.x)),
      Math.max(0, Math.min(img.full_h - 1, p.y)),
    ]);
  if (which === "left") a.fixed = point;
  else a.moving = point;
  if (p.end) {
    const next = pairs.value.map((a) => ({ ...a }));
    pairs.value = draggingPairs!;
    draggingPairs = null;
    await savePairs(next);
  }
}
let pendingMaskAction: Promise<void> | null = null;
function settleMaskAction() { return pendingMaskAction ?? Promise.resolve(); }
function maskAction(action: "confirm" | "reset") {
  if (pendingMaskAction) return pendingMaskAction;
  pendingMaskAction = performMaskAction(action).finally(() => { pendingMaskAction = null; });
  return pendingMaskAction;
}
async function performMaskAction(action: "confirm" | "reset") {
  if (busy.value || props.running) return;
  if (action === "confirm" && previewBusy.value) return;
  const draft = preview.value;
  busy.value = true;
  try {
    await api(`/api/mask/${maskTarget.value || props.current.id}/action`, {
      action,
      draft_token: action === "confirm" ? draft?.token : undefined,
    });
    clearPreview();
    emit("changed");
  } catch (e: any) {
    emit("error", e.message);
  } finally {
    busy.value = false;
  }
}
async function applyAdjust(reset = false) {
  if (busy.value) return;
  const key = resultKey.value,
    id = props.current.id,
    resultId = r.value?.id;
  const values = { ...adj.value };
  busy.value = true;
  try {
    await api(
      `/api/result/${id}/adjust`,
      reset
        ? { reset: true, result_id: resultId }
        : {
            dx: values.dx,
            dy: values.dy,
            rot_deg: values.rot,
            scale: values.scale,
            ref_w: width.value,
            result_id: resultId,
          },
    );
    adjustments[key] = { dx: 0, dy: 0, rot: 0, scale: 1 };
    emit("changed");
  } catch (e: any) {
    emit("error", e.message);
  } finally {
    busy.value = false;
  }
}
defineExpose({ startAnchor, deleteAnchor, cancel, maskAction, fit, draftUndo, cancelInput, settleMaskAction });
</script>
<template>
  <section class="workspace-body">
    <div class="workspace-canvas">
    <div class="context-toolbar" v-if="!isRaw">
      <span>결과 기준: <strong>{{ resultReference }}</strong></span>
      <span v-if="shownResult?.fixed_id !== fixed.id" class="notice">이전 고정 사진과의 결과 · 다시 정합하기 전까지 유지</span>
      <span v-else-if="r?.freshness !== 'current'" class="subtle">입력 변경 전 결과 · 새 기준을 적용하려면 다시 정합하세요</span>
    </div>
    <div class="context-toolbar" v-if="isRaw && tool !== 'adjust'">
      <span>마스크 대상</span
      ><button
        :class="{ on: maskTarget === fixed.id }"
        @click="clearPreview(); maskTarget = fixed.id"
      >
        기준 사진</button
      ><button
        v-if="current.id !== fixed.id"
        :class="{ on: maskTarget === current.id }"
        @click="clearPreview(); maskTarget = current.id"
      >
        현재 사진
      </button>
      <button :disabled="busy || running || previewBusy || !preview" @click="maskAction('confirm')">
        개체 확정 <kbd>Z</kbd></button
      ><button :disabled="busy || running" @click="maskAction('reset')">
        마스크 초기화 <kbd>X</kbd>
      </button>
      <label><input v-model="maskVisible" type="checkbox" />마스크 표시</label
      ><input
        aria-label="마스크 투명도"
        v-model.number="opacity"
        type="range"
        min="0"
        max="1"
        step=".05"
      />
      <span class="subtle">좌클릭 선택 · 우클릭 제외 · Z 확정</span>
    </div>
    <div v-if="isRaw && tool !== 'adjust'" class="anchor-disclosure">
      <button class="anchor-toggle" :aria-expanded="anchorsExpanded" aria-controls="anchor-controls" :aria-label="anchorsExpanded ? '대응점 접기' : '대응점 펼치기'" @click="toggleAnchors">
        <span aria-hidden="true">{{ anchorsExpanded ? '▾' : '▸' }}</span> 대응점 <span class="subtle">선택 기능</span>
      </button>
    <div id="anchor-controls" ref="anchorControls" tabindex="-1" class="context-toolbar" v-show="anchorsExpanded">
      <button
        :disabled="busy || running || current.id === fixed.id"
        @click="startAnchor"
      >
        대응점 선택 <kbd>A</kbd></button
      ><button
        :disabled="busy || running || (!selectedPair && !placing && !candidate)"
        @click="deleteAnchor"
      >
        선택/입력 취소 <kbd>D</kbd>
      </button>
      <label><input type="checkbox" v-model="showAnchors" />점 표시</label>
      <span class="instruction">{{
        placing
          ? pending
            ? "현재 사진을 클릭하고 A로 대응점을 확정하세요"
            : "기준 사진을 클릭하고 A로 선택하세요"
          : "점은 아직 저장되지 않았습니다. A 또는 Z로 용도를 결정하세요"
      }}</span>
      <div class="anchor-list">
        <button
          v-for="(p, i) in pairs"
          :key="p.id"
          :class="{ on: selectedPair === p.id }"
          :title="p.source === 'automatic' ? '자동 추천 앵커 · 두 사진의 같은 위치인지 확인하세요' : '직접 지정한 앵커'"
          @click="selectedPair = p.id"
        >
          {{ i + 1 }}{{ p.source === 'automatic' ? ' 추천' : '' }}{{ p.enabled ? "" : " (제외)" }}
        </button>
      </div>
      <button
        v-if="selectedPair"
        :disabled="running"
        @click="
          savePairs(
            pairs.map((p) =>
              p.id === selectedPair
                ? {
                    ...p,
                    enabled: !(p.requested_enabled ?? p.enabled),
                    requested_enabled: !(p.requested_enabled ?? p.enabled),
                  }
                : p,
            ),
          )
        "
      >
        선택점 사용/제외
      </button>
    </div>
    </div>
    <div
      class="context-toolbar adjustment"
      v-if="tool === 'adjust' && hasResult"
    >
      <label v-for="f in adjFields" :key="f.key"
        >{{ f.title
        }}<input
          type="range"
          v-model.number="adj[f.key]"
          :min="f.min"
          :max="f.max"
          :step="f.step"
          @pointerdown="startGesture"
          @change="finishGesture"
          @keydown="startGesture"
          @keyup="finishGesture" /><input
          type="number"
          v-model.number="adj[f.key]"
          :step="f.step"
          :min="f.min"
          :max="f.max"
          @focus="startGesture"
          @change="finishGesture"
      /></label>
      <button :disabled="busy || !adjDirty || running" @click="applyAdjust()">
        조정 적용</button
      ><button :disabled="busy || running" @click="applyAdjust(true)">
        자동정합 복원</button
      ><span v-if="adjDirty" class="notice">적용 전 · 사진별 임시 보관</span>
    </div>
    <div ref="comparisonArea" class="comparison-area" :class="{ split: side }">
      <PhotoViewport
        v-if="side"
        ref="leftPane"
        :class="{ 'mask-active': isRaw && maskTarget === fixed.id }"
        :src="leftSrc"
        :width="width"
        :height="height"
        :label="`기준 · ${isRaw ? fixed.name : resultReference}${!isRaw && (shownResult?.fixed_id !== fixed.id || r?.freshness !== 'current') ? ' (계산 당시 기준)' : ''}`"
        :view="leftView"
        @update:view="updateView('left', $event)"
        :region-url="leftRegion"
        :interactive="isRaw && tool !== 'adjust'"
        @point="click('left', $event)"
        :points="points('left')"
        @anchor="dragAnchor('left', $event)"
        :overlay="maskOverlay(fixed)"
        :opacity="opacity"
      />
      <PhotoViewport
        v-if="side"
        ref="rightPane"
        :class="{ 'mask-active': isRaw && maskTarget === current.id }"
        :src="rightSrc"
        :width="rightWidth"
        :height="rightHeight"
        :label="`${isRaw ? '현재 사진' : '정합 결과'} · ${current.name}`"
        :view="rightView"
        @update:view="updateView('right', $event)"
        :region-url="rightRegion"
        :interactive="isRaw && tool !== 'adjust'"
        @point="click('right', $event)"
        :points="points('right')"
        @anchor="dragAnchor('right', $event)"
        :overlay="maskOverlay(current)"
        :opacity="opacity"
      />
      <PhotoViewport
        v-else
        ref="leftPane"
        :src="
          mode === 'match' && tool !== 'adjust'
            ? `${base}/match_viz?${version}`
            : leftSrc
        "
        :width="mode === 'match' && tool !== 'adjust' ? matchSize.w : width"
        :height="mode === 'match' && tool !== 'adjust' ? matchSize.h : height"
        :label="`${current.name} · ${tool === 'adjust' ? '미세조정' : mode === 'match' ? '매칭점' : '기준 / 정합 결과'}`"
        :view="leftView"
        @update:view="updateView('left', $event)"
        :region-url="
          mode === 'match' && tool !== 'adjust' ? undefined : leftRegion
        "
        :overlay-region-url="
          mode === 'match' && tool !== 'adjust'
            ? undefined
            : mode === 'false' && tool !== 'adjust'
              ? `${base}/region?kind=false_color&${version}`
              : rightRegion
        "
        :overlay="
          mode === 'match' && tool !== 'adjust'
            ? undefined
            : mode === 'false' && tool !== 'adjust'
              ? `${base}/false_color?${version}`
              : rightSrc
        "
        :opacity="
          tool === 'adjust'
            ? 0.5
            : mode === 'flicker'
              ? flick
                ? 1
                : 0
              : mode === 'false'
                ? opacity
                : 1
        "
        :wipe="mode === 'wipe' && tool !== 'adjust' ? wipe : undefined"
        @update:wipe="wipe = $event"
        :overlay-transform="
          tool === 'adjust'
            ? `translate(${adj.dx}px,${adj.dy}px) rotate(${adj.rot}deg) scale(${adj.scale})`
            : undefined
        "
        :adjustment="tool === 'adjust' && !busy && !running ? adj : undefined"
        @adjust="pointerAdjust"
      />
    </div>
    <div class="viewport-toolbar">
      <button
        v-if="r?.has_previous && !isRaw && tool !== 'adjust'"
        :class="{ on: previous }"
        @click="previous = !previous"
      >
        {{ previous ? "이전 정합 결과 표시 중" : "이전 정합 결과 비교" }}
      </button>
      <button v-if="previous && recoveryAvailable" :disabled="recoveryBlocked" @click="restorePrevious">이전 결과로 돌아가기</button>
      <label
        ><input type="checkbox" v-model="linked" />{{
          isRaw ? "확대 배율 연결" : "확대·위치 연결"
        }}</label
      ><button @click="fit">화면 맞춤 <kbd>0</kbd></button
      >
      <label v-if="!isRaw && mode === 'wipe' && tool !== 'adjust'"
        >비교 경계<input
          aria-label="와이프 경계"
          type="range"
          min="0"
          max="100"
          v-model.number="wipe" /></label
      ><label v-if="!isRaw && mode === 'false'"
        >겹침<input
          aria-label="겹침 투명도"
          type="range"
          min="0"
          max="1"
          step=".05"
          v-model.number="opacity"
      /></label>
      <span class="spacer" /><span class="subtle">{{
        previewBusy ? "마스크 계산 중… 추가 클릭은 최신 선택으로 모아 처리합니다" : busy ? "처리 중…" : "휠 확대 · Space+드래그 이동"
      }}</span>
    </div>
    </div>
    <section v-if="current.id !== fixed.id" class="registration-guidance" aria-labelledby="registration-guidance-title">
      <div class="guidance-heading">
        <strong id="registration-guidance-title">정합 안내</strong>
        <span v-if="recommendation" class="guidance-tag">추천점 검토 중 · 아직 적용하지 않음</span>
        <span v-else-if="r?.validation === 'fit_only'" class="subtle">앵커 적합 오차 · 독립 검증 아님</span>
        <span class="spacer" />
        <button v-if="anchorsExpanded && isRaw" class="primary" :disabled="!recoveryAvailable || recoveryBlocked || enabledPairs < 2" @click="recoverWithAnchors">이 점들로 재정합</button>
        <button v-if="recommendation" :disabled="recoveryBlocked" @click="discardRecommendation">추천 취소</button>
      </div>
      <p role="status" aria-live="polite" aria-atomic="true" :class="{'guidance-error': guidanceError || registrationError}">{{ guidanceText }}</p>
      <p v-if="missingRegions" class="subtle">추천점 부족: {{ missingRegions }}</p>
      <div class="recovery-actions">
        <button :class="{recommended: defaultRecovery === 'mask'}" :disabled="recoveryBlocked" aria-describedby="mask-help" @click="openRecoveryTool('mask')">{{ hasRecoveryMask ? '마스크 재선택' : '마스크 선택' }}</button>
        <button :class="{recommended: defaultRecovery === 'recommend'}" :disabled="!recoveryAvailable || recoveryBlocked" aria-describedby="recommend-help" @click="suggestAnchors">앵커 자동 추천</button>
        <button :disabled="recoveryBlocked" aria-describedby="manual-help" @click="openManualAnchors">앵커 직접 찍기</button>
        <button :disabled="recoveryBlocked || !hasResult" aria-describedby="adjust-help" @click="openRecoveryTool('adjust')">미세조정</button>
      </div>
      <div class="recovery-help">
        <span id="recommend-help" :class="{'default-help': defaultRecovery === 'recommend'}">여러 번 선택한 마스크를 모두 합쳐, 그 안에서 대응점을 고르게 추천합니다. 점을 확인한 뒤 다시 정합하세요.</span>
        <span id="manual-help">두 사진에서 같은 위치를 직접 짚어 정합 기준을 지정하세요.</span>
        <span id="mask-help" :class="{'default-help': defaultRecovery === 'mask'}">{{ hasRecoveryMask ? '기준으로 삼을 영역을 다시 선택한 뒤 정합하세요. 기존 마스크는 유지됩니다.' : '두 사진에서 기준으로 삼을 영역을 선택하고 Z로 확정한 뒤 정합하세요.' }}</span>
        <span id="adjust-help">{{ hasResult ? '위치·회전·균일 배율을 조금씩 조절하세요.' : '미세조정은 정합 결과가 있어야 사용할 수 있습니다.' }}</span>
      </div>
      <details v-if="guidanceDetail && !recommending && !running" class="guidance-detail">
        <summary>결과 설명</summary><p>{{ guidanceDetail }}</p>
      </details>
    </section>
  </section>
</template>
