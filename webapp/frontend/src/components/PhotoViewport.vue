<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { zoomAt, type Viewport } from "../workspace";
type Adjustment = { dx: number; dy: number; rot: number; scale: number };
const props = withDefaults(
  defineProps<{
    src: string;
    width: number;
    height: number;
    label: string;
    view: Viewport;
    regionUrl?: string;
    overlay?: string;
    opacity?: number;
    wipe?: number;
    points?: {
      id: string;
      x: number;
      y: number;
      label: string;
      selected?: boolean;
      disabled?: boolean;
    }[];
    interactive?: boolean;
    overlayTransform?: string;
    overlayRegionUrl?: string;
    adjustment?: Adjustment;
  }>(),
  { opacity: 0.6, points: () => [], interactive: false },
);
const emit = defineEmits<{
  "update:view": [Viewport];
  "update:wipe": [number];
  point: [{ x: number; y: number; button: number }];
  anchor: [{ id: string; x: number; y: number; end: boolean; moved?: boolean }];
  adjust: [Adjustment, "start" | "move" | "end" | "cancel"];
}>();
async function imageLoaded(event: Event) {
  const img = event.currentTarget as HTMLImageElement;
  try {
    await img.decode();
    if (img.isConnected && img.getAttribute('src') === props.src) loading.value = false;
  } catch {
    if (img.isConnected && img.getAttribute('src') === props.src) {
      error.value = true;
      loading.value = false;
    }
  }
}
const host = ref<HTMLElement>(),
  size = ref({ w: 1, h: 1 }),
  space = ref(false),
  error = ref(false),
  loading = ref(true);
const fitScale = computed(
  () =>
    Math.min(size.value.w / props.width, size.value.h / props.height) * 0.94,
);
const scale = computed(() => fitScale.value * props.view.zoom);
const left = computed(
  () => size.value.w / 2 - props.view.cx * props.width * scale.value,
);
const top = computed(
  () => size.value.h / 2 - props.view.cy * props.height * scale.value,
);
const wipeClip = computed(() => {
  if (props.wipe === undefined) return undefined;
  const fraction = (size.value.w * props.wipe / 100 - left.value) / (props.width * scale.value);
  return `inset(0 ${100 * (1 - Math.max(0, Math.min(1, fraction)))}% 0 0)`;
});
let wiping = false;
function moveWipe(e: PointerEvent) {
  if (!wiping) return;
  const x = e.clientX - host.value!.getBoundingClientRect().left;
  emit('update:wipe', Math.max(0, Math.min(100, x / size.value.w * 100)));
}
function startWipe(e: PointerEvent) {
  if (e.button !== 0) return;
  wiping = true;
  const handle = e.currentTarget as HTMLElement;
  handle.focus();
  handle.setPointerCapture(e.pointerId);
  moveWipe(e);
}
function keyWipe(e: KeyboardEvent) {
  if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
  const next = e.code === 'Home' ? 0 : e.code === 'End' ? 100
    : e.code === 'ArrowLeft' ? (props.wipe ?? 50) - (e.shiftKey ? 10 : 1)
    : e.code === 'ArrowRight' ? (props.wipe ?? 50) + (e.shiftKey ? 10 : 1) : null;
  if (next === null) return;
  e.preventDefault();
  e.stopPropagation();
  emit('update:wipe', Math.max(0, Math.min(100, next)));
}
const pictureStyle = computed(() => ({
  width: `${props.width}px`,
  height: `${props.height}px`,
  transform: `translate(${left.value}px,${top.value}px) scale(${scale.value})`,
}));
let resize: ResizeObserver,
  drag: null | {
    x: number;
    y: number;
    ox: number;
    oy: number;
    id?: string;
    pan: boolean;
    button: number;
    moved: boolean;
    adjust?: {
      kind: string;
      start: Adjustment;
      cx: number;
      cy: number;
      angle: number;
      distance: number;
    };
  } = null;
function pos(e: PointerEvent | WheelEvent) {
  const r = host.value!.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}
function imagePos(e: PointerEvent) {
  const p = pos(e);
  return {
    x: (p.x - left.value) / scale.value - 0.5,
    y: (p.y - top.value) / scale.value - 0.5,
  };
}
function down(e: PointerEvent) {
  if (e.button === 2 && !props.interactive) return;
  host.value!.focus();
  host.value!.setPointerCapture(e.pointerId);
  const id =
    (e.target as HTMLElement)
      .closest("[data-anchor]")
      ?.getAttribute("data-anchor") ?? undefined;
  drag = {
    x: e.clientX,
    y: e.clientY,
    ox: e.clientX,
    oy: e.clientY,
    id,
    pan: space.value || e.button === 1 || (!props.interactive && !id),
    button: e.button,
    moved: false,
  };
  if (props.adjustment && !space.value && e.button === 0) {
    const kind =
      (e.target as HTMLElement)
        .closest("[data-adjust]")
        ?.getAttribute("data-adjust") ?? "rotate";
    const bounds = host.value!.getBoundingClientRect();
    const cx =
      bounds.left +
      left.value +
      (props.width / 2 + props.adjustment.dx) * scale.value;
    const cy =
      bounds.top +
      top.value +
      (props.height / 2 + props.adjustment.dy) * scale.value;
    drag.pan = false;
    drag.adjust = {
      kind,
      start: { ...props.adjustment },
      cx,
      cy,
      angle: Math.atan2(e.clientY - cy, e.clientX - cx),
      distance: Math.max(1, Math.hypot(e.clientX - cx, e.clientY - cy)),
    };
    emit("adjust", { ...props.adjustment }, "start");
  }
  e.preventDefault();
}
function move(e: PointerEvent) {
  const p = imagePos(e);
  if (!drag) return;
  if (Math.hypot(e.clientX - drag.ox, e.clientY - drag.oy) > 3)
    drag.moved = true;
  if (drag.adjust) {
    const a = drag.adjust,
      next = { ...a.start };
    if (a.kind === "move") {
      next.dx += (e.clientX - drag.ox) / scale.value;
      next.dy += (e.clientY - drag.oy) / scale.value;
    } else if (a.kind === "scale")
      next.scale = Math.max(
        0.1,
        Math.min(
          5,
          (a.start.scale * Math.hypot(e.clientX - a.cx, e.clientY - a.cy)) /
            a.distance,
        ),
      );
    else
      next.rot +=
        (Math.atan2(
          Math.sin(Math.atan2(e.clientY - a.cy, e.clientX - a.cx) - a.angle),
          Math.cos(Math.atan2(e.clientY - a.cy, e.clientX - a.cx) - a.angle),
        ) *
          180) /
        Math.PI;
    emit("adjust", next, "move");
    return;
  }
  if (drag.pan)
    emit("update:view", {
      ...props.view,
      cx: props.view.cx - (e.clientX - drag.x) / scale.value / props.width,
      cy: props.view.cy - (e.clientY - drag.y) / scale.value / props.height,
    });
  else if (drag.id && drag.moved)
    emit("anchor", { id: drag.id, ...p, end: false, moved: true });
  drag.x = e.clientX;
  drag.y = e.clientY;
}
function up(e: PointerEvent) {
  if (!drag) return;
  const p = imagePos(e),
    d = drag;
  drag = null;
  if (d.adjust) emit("adjust", { ...props.adjustment! }, "end");
  else if (d.id) emit("anchor", { id: d.id, ...p, end: true, moved: d.moved });
  else if (
    !d.pan &&
    !d.moved &&
    p.x >= 0 &&
    p.y >= 0 &&
    p.x <= props.width &&
    p.y <= props.height
  )
    emit("point", { ...p, button: d.button });
  if (host.value?.hasPointerCapture(e.pointerId))
    host.value.releasePointerCapture(e.pointerId);
}
function cancelPointer() {
  if (drag?.adjust) emit("adjust", drag.adjust.start, "cancel");
  drag = null;
}
function wheel(e: WheelEvent) {
  const p = pos(e);
  emit(
    "update:view",
    zoomAt(
      props.view,
      Math.max(
        0.2,
        Math.min(100, props.view.zoom * Math.exp(-e.deltaY * 0.0015)),
      ),
      p.x,
      p.y,
      size.value.w,
      size.value.h,
      props.width,
      props.height,
      fitScale.value,
    ),
  );
}
function fit() {
  emit("update:view", { zoom: 1, cx: 0.5, cy: 0.5 });
}
function onKey(e: KeyboardEvent) {
  if (e.code === "Escape" && drag?.adjust) {
    emit("adjust", drag.adjust.start, "cancel");
    drag = null;
    e.preventDefault();
    return;
  }
  if (e.code === "Space" && !e.repeat) {
    space.value = true;
    e.preventDefault();
  }
}
function releaseKey(e: KeyboardEvent) {
  if (e.code === "Space") space.value = false;
}
const roi = ref<{
  x: number;
  y: number;
  w: number;
  h: number;
  url: string;
} | null>(null);
let timer: ReturnType<typeof setTimeout> | undefined;
function regionURL(
  x: number,
  y: number,
  w: number,
  h: number,
  url = props.regionUrl!,
) {
  return `${url}${url.includes("?") ? "&" : "?"}x=${x}&y=${y}&width=${w}&height=${h}`;
}
function updateROI() {
  clearTimeout(timer);
  roi.value = null;
  if (!props.regionUrl || scale.value < 0.75) return;
  timer = setTimeout(() => {
    const x = Math.max(0, Math.floor(-left.value / scale.value)),
      y = Math.max(0, Math.floor(-top.value / scale.value));
    const w = Math.min(
        2048,
        props.width - x,
        Math.ceil(size.value.w / scale.value) + 2,
      ),
      h = Math.min(
        2048,
        props.height - y,
        Math.ceil(size.value.h / scale.value) + 2,
      );
    if (w > 0 && h > 0) roi.value = { x, y, w, h, url: regionURL(x, y, w, h) };
  }, 100);
}
watch(
  [
    () => props.view.zoom,
    () => props.view.cx,
    () => props.view.cy,
    () => props.regionUrl,
    size,
  ],
  updateROI,
);
watch(
  () => props.src,
  () => {
    error.value = false;
    loading.value = true;
    updateROI();
  },
);
onMounted(() => {
  resize = new ResizeObserver((es) => {
    const r = es[0].contentRect;
    size.value = { w: r.width, h: r.height };
  });
  resize.observe(host.value!);
  window.addEventListener("keyup", releaseKey);
});
onUnmounted(() => {
  resize?.disconnect();
  clearTimeout(timer);
  window.removeEventListener("keyup", releaseKey);
});
defineExpose({ fit });
</script>
<template>
  <div
    ref="host"
    class="photo-viewport"
    :class="{ placing: interactive && !space }"
    tabindex="0"
    :aria-label="label"
    :aria-busy="loading"
    @wheel.prevent="wheel"
    @pointerdown="down"
    @pointermove="move"
    @pointerup="up"
    @pointercancel="cancelPointer"
    @keydown="onKey"
    @blur="space = false"
    @contextmenu.prevent
  >
    <div class="viewport-caption">
      {{ label }} <span>{{ Math.round(scale * 100) }}%</span>
    </div>
    <div class="photo-plane" :style="pictureStyle">
      <img
        :key="src"
        class="photo-layer"
        :src="src"
        draggable="false"
        decoding="async"
        @load="imageLoaded"
        @error="error = true; loading = false"
      />
      <img
        v-if="roi"
        :key="roi.url"
        class="roi-layer"
        :src="roi.url"
        :style="{
          left: roi.x + 'px',
          top: roi.y + 'px',
          width: roi.w + 'px',
          height: roi.h + 'px',
        }"
        draggable="false"
      />
      <img
        v-if="overlay"
        :key="`${src}:${overlay}`"
        class="photo-layer"
        :src="overlay"
        draggable="false"
        :style="{
          opacity,
          clipPath: wipeClip,
          transform: overlayTransform,
        }"
      />
      <div
        v-if="roi && overlayRegionUrl"
        class="photo-layer"
        :style="{
          opacity,
          clipPath: wipeClip,
          transform: overlayTransform,
        }"
      >
        <img
          :key="regionURL(roi.x, roi.y, roi.w, roi.h, overlayRegionUrl)"
          class="roi-layer"
          :src="regionURL(roi.x, roi.y, roi.w, roi.h, overlayRegionUrl)"
          :style="{
            left: roi.x + 'px',
            top: roi.y + 'px',
            width: roi.w + 'px',
            height: roi.h + 'px',
          }"
        />
      </div>
      <slot :scale="scale" />
      <div
        v-if="adjustment"
        class="adjust-frame"
        data-adjust="move"
        :style="{ transform: overlayTransform, borderWidth: 1 / scale + 'px' }"
      >
        <button
          v-for="corner in ['nw', 'ne', 'se', 'sw']"
          :key="corner"
          data-adjust="scale"
          class="adjust-handle"
          :class="corner"
          :aria-label="`${corner} 균일 크기 조정`"
          :style="{
            transform: `translate(-50%,-50%) scale(${1 / scale / adjustment.scale})`,
          }"
        />
        <button
          data-adjust="rotate"
          class="adjust-handle rotate"
          aria-label="회전 핸들"
          :style="{
            transform: `translate(-50%,-50%) scale(${1 / scale / adjustment.scale})`,
          }"
        >
          ↻
        </button>
      </div>
      <button
        v-for="p in points"
        :key="p.id"
        class="anchor-dot"
        :class="{ selected: p.selected, disabled: p.disabled }"
        :data-anchor="p.id"
        :aria-label="`대응점 ${p.label}`"
        :style="{
          left: p.x + 0.5 + 'px',
          top: p.y + 0.5 + 'px',
          transform: `translate(-50%,-50%) scale(${1 / scale})`,
        }"
      >
        {{ p.label }}
      </button>
    </div>
    <div v-if="wipe !== undefined && overlay" class="wipe-divider" :style="{ left: wipe + '%' }">
      <button class="wipe-handle" role="slider" aria-label="와이프 경계 이동"
        aria-orientation="horizontal" aria-valuemin="0" aria-valuemax="100" :aria-valuenow="Math.round(wipe)"
        :style="{ transform: `translateX(${wipe < 3 ? 0 : wipe > 97 ? -100 : -50}%)` }"
        @pointerdown.stop.prevent="startWipe" @pointermove.stop="moveWipe"
        @pointerup.stop="wiping = false" @pointercancel.stop="wiping = false" @lostpointercapture="wiping = false"
        @keydown="keyWipe" @dblclick.stop="emit('update:wipe', 50)">
        <span aria-hidden="true">↔</span>
      </button>
    </div>
    <div v-if="loading" class="viewport-loading" role="status">사진 불러오는 중…</div>
    <div v-if="error" class="viewport-error">
      사진을 불러오지 못했습니다. 다시 선택해 주세요.
    </div>
  </div>
</template>
