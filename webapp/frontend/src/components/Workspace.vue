<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from "vue";
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
}>();
const emit = defineEmits<{
  changed: [];
  error: [string];
  "update:tool": [Tool];
}>();
const leftPane = ref<InstanceType<typeof PhotoViewport>>(),
  rightPane = ref<InstanceType<typeof PhotoViewport>>();
const linked = ref(true),
  loupe = ref(false),
  maskVisible = ref(true),
  opacity = ref(0.65),
  wipe = ref(50),
  busy = ref(false),
  maskTarget = ref(""),
  hover = ref<{ x: number; y: number } | null>(null);
const pairs = ref<Anchor[]>([]),
  pairRevision = ref(0),
  selectedPair = ref<string | null>(null),
  placing = ref(false),
  pending = ref<Point | null>(null),
  showAnchors = ref(true);
const candidate = ref<{which: "left" | "right"; x: number; y: number} | null>(null);
const preview = ref<{token: string; overlay: string; imageId: string} | null>(null);
const previewBusy = ref(false);
let previewSequence = 0;
let previewPoints: {x:number; y:number; label:number}[] = [];
let previewImage = "";
function clearPreview() {
  previewSequence++;
  candidate.value = null;
  preview.value = null;
  previewBusy.value = false;
  previewPoints = [];
  previewImage = "";
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
    !hasResult.value ||
    props.fixed.id === props.current.id,
);
const resultKey = computed(
  () => `${props.fixed.id}:${props.current.id}:${r.value?.id ?? ""}`,
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
  () => `v=${props.revision}&result=${r.value?.id ?? ""}`,
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
function actual() {
  leftPane.value?.actual();
  rightPane.value?.actual();
}
let anchorRequest = 0;
async function loadAnchors() {
  const token = ++anchorRequest;
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
    placing.value = false;
    pending.value = null;
    selectedPair.value = null;
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
  if (props.current.id === props.fixed.id) return;
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
async function savePairs(next: Anchor[]) {
  if (busy.value || props.running) return false;
  busy.value = true;
  const key = pairKey.value,
    mid = props.current.id,
    fid = props.fixed.id,
    rev = pairRevision.value;
  try {
    await api(
      `/api/anchors/${mid}`,
      { pairs: next, base_revision: rev, fixed_id: fid },
      "PUT",
    );
    if (key === pairKey.value) await loadAnchors();
    emit("changed");
    return key === pairKey.value;
  } catch (e: any) {
    emit("error", e.message);
    if (key === pairKey.value) await loadAnchors();
    return false;
  } finally {
    busy.value = false;
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
    if (previewImage !== img.id) previewPoints = [];
    previewImage = img.id;
    previewPoints.push({
      x: Math.max(0, Math.min(img.w - 1, ((p.x + .5) * img.w) / img.full_w - .5)),
      y: Math.max(0, Math.min(img.h - 1, ((p.y + .5) * img.h) / img.full_h - .5)),
      label: p.button === 2 ? 0 : 1,
    });
    const sequence = ++previewSequence;
    previewBusy.value = true;
    preview.value = null;
    try {
      const result = await api(`/api/mask/${img.id}/preview`, {points: [...previewPoints]});
      if (sequence === previewSequence) preview.value = {...result, imageId: img.id};
    } catch (e: any) {
      if (sequence === previewSequence) emit("error", e.message);
    } finally {
      if (sequence === previewSequence) previewBusy.value = false;
    }
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
  if (!isRaw.value || !showAnchors.value) return [];
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
      Math.max(0, Math.min(img.full_w, p.x)),
      Math.max(0, Math.min(img.full_h, p.y)),
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
async function maskAction(action: "confirm" | "reset") {
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
defineExpose({ startAnchor, deleteAnchor, cancel, maskAction, fit, draftUndo, cancelInput });
</script>
<template>
  <section class="workspace-body">
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
      <span class="subtle">클릭 → A 대응점 / Z 마스크 · 우클릭 제외</span>
    </div>
    <div class="context-toolbar" v-if="isRaw && tool !== 'adjust'">
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
          @click="selectedPair = p.id"
        >
          {{ i + 1 }}{{ p.enabled ? "" : " (제외)" }}
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
    <div class="comparison-area" :class="{ split: side }">
      <PhotoViewport
        v-if="side"
        ref="leftPane"
        :src="leftSrc"
        :width="width"
        :height="height"
        :label="`기준 · ${fixed.name}${!isRaw && r?.freshness !== 'current' ? ' (계산 당시 기준)' : ''}`"
        :view="leftView"
        @update:view="updateView('left', $event)"
        :region-url="leftRegion"
        :interactive="isRaw && tool !== 'adjust'"
        @point="click('left', $event)"
        :points="points('left')"
        @anchor="dragAnchor('left', $event)"
        :overlay="
          maskVisible
            ? preview?.imageId === fixed.id ? preview.overlay : `/api/mask/${fixed.id}/overlay?v=${revision}`
            : undefined
        "
        :opacity="opacity"
        :loupe="loupe"
        :cursor-point="!isRaw && linked ? hover : null"
        @hover="hover = $event"
      />
      <PhotoViewport
        v-if="side"
        ref="rightPane"
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
        :overlay="
          maskVisible
            ? preview?.imageId === current.id ? preview.overlay : `/api/mask/${current.id}/overlay?v=${revision}`
            : undefined
        "
        :opacity="opacity"
        :loupe="loupe"
        :cursor-point="!isRaw && linked ? hover : null"
        @hover="hover = $event"
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
        :overlay-transform="
          tool === 'adjust'
            ? `translate(${adj.dx}px,${adj.dy}px) rotate(${adj.rot}deg) scale(${adj.scale})`
            : undefined
        "
        :adjustment="tool === 'adjust' && !busy && !running ? adj : undefined"
        @adjust="pointerAdjust"
        :loupe="loupe"
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
      <label
        ><input type="checkbox" v-model="linked" />{{
          isRaw ? "확대 배율 연결" : "확대·위치 연결"
        }}</label
      ><button @click="fit">화면 맞춤 <kbd>0</kbd></button
      ><button @click="actual">100%</button
      ><button :class="{ on: loupe }" @click="loupe = !loupe">부분 확대</button>
      <label v-if="!isRaw && mode === 'wipe'"
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
        previewBusy ? "마스크 미리보기 생성 중… 첫 사용은 모델 다운로드가 필요합니다. A 선택 가능" : busy ? "처리 중…" : "휠 확대 · Space+드래그 이동"
      }}</span>
    </div>
  </section>
</template>
