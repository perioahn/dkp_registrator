<script setup lang="ts">
import { ref } from "vue";
import type { Edits } from "../photoEdits";
defineProps<{
  edits: Edits;
  mode: "view" | "crop";
  cropRatio: number | null;
  straighten: boolean;
  disabled?: boolean;
}>();
const emit = defineEmits<{
  change: [
    key: "brightness" | "contrast" | "rot90" | "flipH" | "flipV",
    value: number | boolean,
  ];
  fine: [value: number];
  crop: [];
  apply: [];
  cancel: [];
  clear: [];
  ratio: [value: number | null];
  straighten: [];
  reset: [];
  gesture: [active: boolean];
}>();
const customW = ref(1),
  customH = ref(1);
const presets: [string, number][] = [
  ["1:1", 1],
  ["4:3", 4 / 3],
  ["3:2", 3 / 2],
  ["16:10", 1.6],
  ["16:9", 16 / 9],
];
function number(key: "brightness" | "contrast" | "fineDeg", event: Event) {
  const value = (event.target as HTMLInputElement).valueAsNumber;
  if (!Number.isFinite(value)) return;
  if (key === "fineDeg") emit("fine", Math.max(-15, Math.min(15, value)));
  else emit("change", key, Math.max(-100, Math.min(100, value)));
}
function custom() {
  if (customW.value > 0 && customH.value > 0)
    emit("ratio", customW.value / customH.value);
}
</script>
<template>
  <div class="editor-controls">
    <section class="control-section">
      <div class="section-heading">
        <h3>빛과 톤</h3>
        <span>밝기 · 대비</span>
      </div>
      <label
        v-for="[label, key] in [
          ['밝기', 'brightness'],
          ['대비', 'contrast'],
        ] as const"
        :key="key"
        class="adjust-control"
      >
        <span
          >{{ label
          }}<button
            :disabled="disabled"
            class="reset-value"
            :aria-label="`${label} 초기화`"
            @click.prevent="emit('change', key, 0)"
          >
            초기화
          </button></span
        >
        <div class="slider-value">
          <input
            type="range"
            :aria-label="label"
            :value="edits[key]"
            min="-100"
            max="100"
            :disabled="disabled"
            @pointerdown="emit('gesture', true)"
            @pointerup="emit('gesture', false)"
            @pointercancel="emit('gesture', false)"
            @change="emit('gesture', false)"
            @blur="emit('gesture', false)"
            @input="number(key, $event)"
          />
          <input
            type="number"
            :aria-label="`${label} 수치`"
            :value="edits[key]"
            min="-100"
            max="100"
            :disabled="disabled"
            @focus="emit('gesture', true)"
            @blur="emit('gesture', false)"
            @input="number(key, $event)"
          />
        </div>
      </label>
    </section>
    <section class="control-section">
      <div class="section-heading">
        <h3>구도</h3>
        <span>크롭 · 회전</span>
      </div>
      <button
        v-if="mode === 'view'"
        class="wide crop-open"
        :disabled="disabled"
        @click="emit('crop')"
      >
        ✂ 크롭
      </button>
      <div class="control-grid">
        <button
          :disabled="disabled"
          title="90° 반시계"
          @click="emit('change', 'rot90', (edits.rot90 + 3) % 4)"
        >
          ↶ 90°
        </button>
        <button
          :disabled="disabled"
          title="90° 시계"
          @click="emit('change', 'rot90', (edits.rot90 + 1) % 4)"
        >
          ↷ 90°
        </button>
        <button
          :disabled="disabled"
          :aria-pressed="edits.flipH"
          @click="emit('change', 'flipH', !edits.flipH)"
        >
          좌우 반전
        </button>
        <button
          :disabled="disabled"
          :aria-pressed="edits.flipV"
          @click="emit('change', 'flipV', !edits.flipV)"
        >
          상하 반전
        </button>
      </div>
      <label class="adjust-control"
        ><span
          >미세 회전<button
            class="reset-value"
            :disabled="disabled"
            @click.prevent="emit('fine', 0)"
          >
            초기화
          </button></span
        >
        <div class="slider-value">
          <input
            type="range"
            aria-label="미세 회전"
            :value="edits.fineDeg"
            min="-15"
            max="15"
            step=".1"
            :disabled="disabled"
            @pointerdown="emit('gesture', true)"
            @pointerup="emit('gesture', false)"
            @pointercancel="emit('gesture', false)"
            @change="emit('gesture', false)"
            @blur="emit('gesture', false)"
            @input="number('fineDeg', $event)"
          />
          <input
            type="number"
            aria-label="미세 회전 수치"
            :value="edits.fineDeg"
            min="-15"
            max="15"
            step=".1"
            :disabled="disabled"
            @focus="emit('gesture', true)"
            @blur="emit('gesture', false)"
            @input="number('fineDeg', $event)"
          />
        </div>
      </label>
      <button
        class="wide"
        :aria-pressed="straighten"
        :disabled="disabled"
        @click="emit('straighten')"
      >
        수평선 긋기
      </button>
      <div v-if="mode === 'crop'" class="crop-settings">
        <p>영역을 지정하고 적용하세요. <kbd>Enter</kbd></p>
        <div class="ratio-grid">
          <button
            :disabled="disabled"
            :aria-pressed="cropRatio === null"
            @click="emit('ratio', null)"
          >
            자유</button
          ><button
            v-for="[label, value] in presets"
            :key="label"
            :disabled="disabled"
            :aria-pressed="cropRatio === value"
            @click="emit('ratio', value)"
          >
            {{ label }}
          </button>
        </div>
        <div class="custom-ratio">
          <input
            v-model.number="customW"
            aria-label="크롭 가로 비율"
            type="number"
            min=".01"
            :disabled="disabled"
            @change="custom"
          />
          :
          <input
            v-model.number="customH"
            aria-label="크롭 세로 비율"
            type="number"
            min=".01"
            :disabled="disabled"
            @change="custom"
          />
        </div>
        <div class="control-grid">
          <button :disabled="disabled" @click="emit('cancel')">취소</button
          ><button class="primary" :disabled="disabled" @click="emit('apply')">
            ✓ 적용
          </button>
        </div>
        <button
          v-if="edits.crop"
          class="wide"
          :disabled="disabled"
          @click="emit('clear')"
        >
          크롭 제거
        </button>
      </div>
    </section>
    <button class="wide reset-all" :disabled="disabled" @click="emit('reset')">
      모든 보정 초기화
    </button>
    <p class="controls-help">
      휠로 확대 · Space+드래그로 이동<br />크롭은 영역을 자르며 비율을 늘이지
      않습니다.
    </p>
  </div>
</template>
<style scoped>
.editor-controls {
  color: #e2e8ef;
  font:
    13px/1.5 "Segoe UI",
    "Malgun Gothic",
    sans-serif;
  min-width: 0;
}
.control-section {
  padding: 16px 0;
  border-bottom: 1px solid #35404c;
}
.control-section:first-child {
  padding-top: 0;
}
.section-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 14px;
}
.section-heading h3 {
  font-size: 14px;
  font-weight: 600;
  margin: 0;
}
.section-heading > span,
.controls-help {
  font-size: 11px;
  color: #94a3b4;
}
.adjust-control {
  display: block;
  margin: 14px 0;
}
.adjust-control > span {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.slider-value {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 5px;
}
input[type="range"] {
  width: 0;
  flex: 1;
  accent-color: #a7d9c5;
  cursor: pointer;
}
input[type="number"] {
  width: 62px;
  background: #141b22;
  border: 1px solid #44525e;
  border-radius: 5px;
  padding: 6px;
  color: inherit;
  font: inherit;
  min-width: 0;
}
.control-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 7px;
}
.ratio-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
}
.custom-ratio {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
  margin: 10px 0;
}
.custom-ratio input {
  width: 70px;
}
button {
  background: #28323d;
  border: 1px solid #465461;
  color: inherit;
  border-radius: 6px;
  padding: 6px 10px;
  min-height: 34px;
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
button:hover {
  border-color: #a7d9c5;
}
button:disabled {
  opacity: 0.4;
  cursor: default;
}
button[aria-pressed="true"] {
  background: #29483f;
  border-color: #a7d9c5;
}
.primary {
  background: #acd8c6;
  color: #172c26;
  border-color: #acd8c6;
}
.wide {
  width: 100%;
  margin-top: 8px;
}
.reset-value {
  border: 0;
  background: transparent;
  color: #9eb0bf;
  min-height: 24px;
  padding: 2px 4px;
  font-size: 10px;
}
.reset-all {
  margin-top: 16px;
}
.controls-help {
  line-height: 1.8;
  margin: 12px 0 0;
}
.crop-settings p {
  font-size: 12px;
  color: #bacbd7;
}
.crop-settings {
  margin-top: 12px;
}
button:focus-visible,
input:focus-visible {
  outline: 2px solid #a7d9c5;
  outline-offset: 2px;
}
</style>
