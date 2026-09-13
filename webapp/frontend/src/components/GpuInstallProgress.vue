<script setup lang="ts">
defineProps<{ gpu: any }>();
const stages: Record<string, string> = {url: '다운로드 준비', download: '다운로드', extract: '압축 해제', finalize: '설치 마무리', done: '설치 완료', error: '설치 실패'};
const mb = (n: number) => ((n || 0) / 1048576).toFixed(1);
</script>

<template>
  <div v-if="gpu?.installing || gpu?.phase === 'done' || gpu?.error" class="gpu-install">
    <strong>{{ stages[gpu.phase] || '설치 준비' }}<template v-if="gpu.installing && gpu.pkg"> · {{ gpu.pkg }} ({{ gpu.pkg === 'torch' ? 1 : 2 }}/2)</template></strong>
    <template v-if="gpu.installing">
      <progress aria-label="현재 설치 단계 진행률" :value="gpu.total > 0 ? gpu.done : undefined" :max="gpu.total > 0 ? gpu.total : 1"></progress>
      <p v-if="gpu.total > 0">{{ Math.min(100, Math.round(gpu.done / gpu.total * 100)) }}% · {{ mb(gpu.done) }} / {{ mb(gpu.total) }} MB</p>
      <p v-else>진행 중…<template v-if="gpu.done"> {{ mb(gpu.done) }} MB</template></p>
    </template>
    <p v-if="gpu.install_dir">설치 위치<br /><span>{{ gpu.install_dir }}</span></p>
    <p v-if="gpu.phase === 'done'">앱을 다시 시작하면 GPU 가속이 적용됩니다.</p>
  </div>
</template>

<style scoped>
.gpu-install { margin-top: 10px; padding: 10px; background: #eef6fb; border-radius: 6px; font-size: 12px; }
progress { width: 100%; margin-top: 8px; accent-color: #217aa5; }
p { margin: 5px 0; }
span { overflow-wrap: anywhere; user-select: text; }
</style>
