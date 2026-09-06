// 결과 보기 방식 — Moving 전환/재생성에도 유지되는 모듈 스코프 상태
import { ref } from "vue";
import { reactive } from "vue";
import type { Viewport } from "./workspace";
export const views = reactive<Record<string, Viewport>>({});
export const adjustments = reactive<
  Record<string, { dx: number; dy: number; rot: number; scale: number }>
>({});
export function viewFor(key: string) {
  return views[key] ?? (views[key] = { zoom: 1, cx: 0.5, cy: 0.5 });
}

export type ViewMode =
  "wipe" | "false" | "flicker" | "side" | "match" | "adjust";
export const mode = ref<ViewMode>("wipe");
export const wipe = ref(50);
export const fcOpacity = ref(70);
