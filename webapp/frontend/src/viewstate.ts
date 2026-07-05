// 결과 보기 방식 — Moving 전환/재생성에도 유지되는 모듈 스코프 상태
import { ref } from 'vue'

export type ViewMode = 'wipe' | 'false' | 'flicker' | 'side' | 'match'
export const mode = ref<ViewMode>('wipe')
export const wipe = ref(50)
export const fcOpacity = ref(70)
