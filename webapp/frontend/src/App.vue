<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from "vue";
import Workspace from "./components/Workspace.vue";
import FixedEditor from "./components/FixedEditor.vue";
import GpuInstallProgress from "./components/GpuInstallProgress.vue";
import { prefetchEditorPreview } from "./editorPreview";
import {
  api,
  preferredResult,
  completionSelection,
  nextReview,
  shortcut,
  type Photo,
  type Tool,
  type ComparisonMode,
  type LocalJob,
} from "./workspace";

const photos = ref<Photo[]>([]),
  appVersion = ref(""),
  recoveryAvailable = ref(false),
  fixedId = ref<string | null>(null),
  activeId = ref<string | null>(null),
  checked = ref(new Set<string>()),
  revision = ref(0);
const running = ref(false),
  starting = ref(false),
  uploading = ref(false),
  mutating = ref(false),
  recoveryBusy = ref(false),
  message = ref("사진을 추가하면 첫 사진이 기준이 됩니다."),
  error = ref("");
const tool = ref<Tool>("compare"),
  mode = ref<ComparisonMode>("side"),
  inspector = ref(true),
  focused = ref(false),
  connectionLost = ref(false),
  listSide = ref(localStorage.getItem("dkp-list-side") === "true"),
  search = ref(""),
  filter = ref("all");
const history = ref({ undo_label: "", redo_label: "" }),
  lazy = ref(false),
  job = ref<any>(null),
  localJob = ref<LocalJob | null>(null),
  completedJob = ref<LocalJob | null>(null);
const fileInput = ref<HTMLInputElement>(),
  workspace = ref<InstanceType<typeof Workspace>>(),
  surface = ref<HTMLElement>(),
  gpu = ref<any>(null);
const saving = ref(false), selectingFolder = ref(false), switchingDevice = ref(false);
const exportMode = ref('download'), saveDir = ref(''), exportNotice = ref('');
const cpuHelp = computed(() => {
  if (gpu.value?.mode === 'cpu' && gpu.value?.accelerator)
    return '테스트를 위해 CPU를 사용 중입니다. 아래 GPU 가속 버튼으로 복귀할 수 있습니다.';
  if (gpu.value?.platform === 'darwin')
    return 'Metal 가속을 사용할 수 없는 환경입니다. macOS와 앱 업데이트를 확인하세요. 지원되지 않는 Mac에서는 CPU로 계속 사용할 수 있습니다.';
  if (gpu.value?.gpu_name)
    return gpu.value?.frozen && !gpu.value?.installed
      ? 'NVIDIA GPU가 감지됐습니다. GPU 가속을 설치한 뒤 앱을 다시 실행하세요. 다운로드가 필요합니다.'
      : 'NVIDIA 드라이버와 CUDA 지원 실행 환경을 확인한 뒤 앱을 다시 실행하세요. 지금은 CPU로 사용할 수 있습니다.';
  return 'GPU 가속이 감지되지 않았습니다. NVIDIA GPU가 있는 PC라면 그래픽 드라이버와 GPU 가속 설치 상태를 확인한 뒤 앱을 다시 실행하세요. 그 외에는 CPU로 계속 사용할 수 있으며 계산이 더 오래 걸릴 수 있습니다.';
});
let navigation = 0,
  requestSequence = 0,
  appliedSequence = 0,
  reviewQueue: string[] = [],
  eventSource: EventSource,
  doneDuringStart: any = null;
const fixed = computed(
  () => photos.value.find((i) => i.id === fixedId.value) ?? null,
);
watch(
  () => fixed.value && `${fixed.value.id}:${fixed.value.revision}`,
  () => {
    if (fixed.value) prefetchEditorPreview(fixed.value);
  },
);
const active = computed(
  () => photos.value.find((i) => i.id === activeId.value) ?? null,
);
const photoNames = new Intl.Collator('ko', {numeric: true, sensitivity: 'base'});
const photoOrder = ref<string[]>((() => {
  try {
    const saved = JSON.parse(localStorage.getItem('dkp-photo-order') ?? '[]');
    return Array.isArray(saved) ? saved.filter(id => typeof id === 'string') : [];
  } catch { return []; }
})());
const orderedPhotos = computed(() => {
  const ranks = new Map(photoOrder.value.map((id,i) => [id,i]));
  return [...photos.value].sort((a,b) =>
    (ranks.get(a.id) ?? Infinity) - (ranks.get(b.id) ?? Infinity) || photoNames.compare(a.name,b.name));
});
watch(photoOrder, ids => localStorage.setItem('dkp-photo-order',JSON.stringify(ids)));
const fixedDropActive = ref(false);
const draggingPhoto = ref<string | null>(null);
const insertion = ref<{id: string; after: boolean} | null>(null);
function dragPhoto(e: DragEvent, id: string) {
  e.dataTransfer?.clearData();
  e.dataTransfer?.setData('application/x-dkp-photo', id);
  if (e.dataTransfer) e.dataTransfer.effectAllowed = 'move';
  draggingPhoto.value = id;
}
function endPhotoDrag() {
  draggingPhoto.value = null;
  insertion.value = null;
  fixedDropActive.value = false;
}
function previewOrder(e: DragEvent, id: string) {
  if (!draggingPhoto.value) {
    if (Array.from(e.dataTransfer?.types ?? []).includes('Files')) e.preventDefault();
    return;
  }
  if (draggingPhoto.value === id || tool.value === 'edit') return;
  e.preventDefault();
  const box = (e.currentTarget as HTMLElement).getBoundingClientRect();
  insertion.value = {id, after: listSide.value ? e.clientY > box.y+box.height/2 : e.clientX > box.x+box.width/2};
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
}
function dropPhoto(e: DragEvent, id: string) {
  const source = e.dataTransfer?.getData('application/x-dkp-photo');
  if (!source) { dropFiles(e); return; }
  if (tool.value !== 'edit' && source !== id && photos.value.some(p => p.id === source)) {
    const ids = orderedPhotos.value.map(p => p.id).filter(p => p !== source);
    const at = ids.indexOf(id);
    if (at >= 0) {
      ids.splice(at + (insertion.value?.id === id && insertion.value.after ? 1 : 0),0,source);
      photoOrder.value = ids;
      reviewQueue = [];
      message.value = '사진 순서를 변경했습니다.';
    }
  }
  endPhotoDrag();
}
function dropFiles(e: DragEvent) {
  const types = Array.from(e.dataTransfer?.types ?? []);
  if (draggingPhoto.value || types.some(t => ['application/x-dkp-photo','text/uri-list','text/html'].includes(t))) return;
  upload(e.dataTransfer?.files ?? null);
}
function guardNativeDrag(e: DragEvent) {
  if (!(e.target as HTMLElement).closest('.photo-card')) e.preventDefault();
}
function dropFixed(e: DragEvent) {
  fixedDropActive.value = false;
  const id = e.dataTransfer?.getData('application/x-dkp-photo');
  if (id && photos.value.some(p => p.id === id)) changeFixed(id);
  else if (!id) dropFiles(e);
  endPhotoDrag();
}
const moving = computed(() =>
  orderedPhotos.value.filter((i) => i.id !== fixedId.value),
);
const filtered = computed(() =>
  orderedPhotos.value.filter(
    (p) =>
      p.name.toLocaleLowerCase().includes(search.value.toLocaleLowerCase()) &&
      (filter.value === "all" ||
        (p.id !== fixedId.value &&
          (filter.value === "unregistered"
            ? !p.result
            : filter.value === "failed"
              ? p.result?.status === "fail" || p.result?.latest_attempt_failed
              : filter.value === "confirmed"
                ? p.result?.review_status === "confirmed"
                : p.result?.review_status !== "confirmed" ||
                  p.result?.freshness !== "current"))),
  ),
);
const checkedMoving = computed(() =>
  moving.value.filter((p) => checked.value.has(p.id)).map((p) => p.id),
);
const resultIds = computed(() =>
  moving.value
    .filter((p) => p.result && p.result.status !== "fail")
    .map((p) => p.id),
);
const jobSaveIds = computed(() => {
  const done = completedJob.value;
  if (!done || done.fixedId !== fixedId.value) return [];
  return done.targets.filter(
    (id) =>
      done.resultVersions?.[id] &&
      photos.value.find((p) => p.id === id)?.result?.id ===
        done.resultVersions[id],
  );
});
const confirmedIds = computed(() =>
  moving.value
    .filter(
      (p) =>
        p.result?.review_status === "confirmed" &&
        p.result?.freshness === "current",
    )
    .map((p) => p.id),
);
const canRun = computed(
  () =>
    tool.value !== "edit" &&
    !running.value &&
    !starting.value &&
    !mutating.value &&
    !uploading.value &&
    !recoveryBusy.value &&
    moving.value.length > 0,
);
const currentIsMoving = computed(
  () => !!active.value && active.value.id !== fixedId.value,
);
const result = computed(() => active.value?.result);
const reviewable = computed(
  () =>
    result.value &&
    result.value.status !== "fail" &&
    result.value.freshness === "current" &&
    !running.value,
);
const toolNames: { key: Tool; name: string }[] = [
  { key: "compare", name: "비교" },
  { key: "mask", name: "마스크" },
  { key: "adjust", name: "미세조정" },
];
const modeNames: { key: ComparisonMode; name: string }[] = [
  { key: "side", name: "나란히" },
  { key: "wipe", name: "와이프" },
  { key: "false", name: "색상 겹침" },
  { key: "flicker", name: "깜빡임" },
  { key: "match", name: "매칭점" },
];
watch(listSide, (v) => localStorage.setItem("dkp-list-side", String(v)));
watch([search, filter], () => {
  reviewQueue = [];
});
function report(e: any) {
  error.value = e?.message ?? String(e);
}
let refreshFlight: Promise<void> | null = null;
let refreshQueued = false;
function refresh(): Promise<void> {
  refreshQueued = true;
  if (refreshFlight) return refreshFlight;
  refreshFlight = (async () => {
    do {
      refreshQueued = false;
      await refreshState();
    } while (refreshQueued);
  })().finally(() => { refreshFlight = null; });
  return refreshFlight;
}
async function refreshState() {
  const seq = ++requestSequence;
  try {
    const d = await api("/api/state");
    if (seq < appliedSequence) return;
    appliedSequence = seq;
    photos.value = d.images;
    photoOrder.value = photoOrder.value.filter(id => d.images.some((p: Photo) => p.id === id));
    appVersion.value = d.version ?? "구버전 서버";
    recoveryAvailable.value = !!d.capabilities?.anchor_recovery;
    revision.value = d.revision ?? 0;
    running.value = d.running;
    fixedId.value =
      d.fixed_id ?? d.images.find((p: Photo) => p.role === "fixed")?.id ?? null;
    history.value = d.history ?? { undo_label: "", redo_label: "" };
    job.value = d.job;
    const available = new Set(d.images.map((p: Photo) => p.id));
    checked.value = new Set(
      [...checked.value].filter(
        (id) => available.has(id) && id !== fixedId.value,
      ),
    );
    if (!activeId.value || !available.has(activeId.value))
      activeId.value =
        d.images.find((p: Photo) => p.id !== fixedId.value)?.id ??
        fixedId.value;
  } catch (e) {
    report(e);
  }
}
function focusCanvas() {
  nextTick(() => surface.value?.focus());
}
function toggleFocus() {
  focused.value = !focused.value;
  focusCanvas();
}
function exitFocus(e: KeyboardEvent) {
  if (focused.value && e.code === 'Escape' && !e.isComposing && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
    focused.value = false;
    e.preventDefault();
    e.stopPropagation();
  }
}
function select(id: string, user = true) {
  if (user) {
    navigation++;
    reviewQueue = [];
  }
  activeId.value = id;
  error.value = "";
  focusCanvas();
}
function chooseTool(value: Tool) {
  navigation++;
  error.value = '';
  tool.value = value;
  focusCanvas();
}
function chooseMode(value: ComparisonMode) {
  navigation++;
  mode.value = value;
  focusCanvas();
}
function toggle(id: string) {
  const n = new Set(checked.value);
  n.has(id) ? n.delete(id) : n.add(id);
  checked.value = n;
}
async function upload(files: FileList | File[] | null) {
  if (!files?.length || uploading.value || running.value) return;
  uploading.value = true;
  error.value = "";
  try {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const res = await fetch("/api/upload", { method: "POST", body: form }),
      d = await res.json();
    if (!res.ok) throw Error(d.detail ?? "사진 추가 실패");
    await refresh();
    const bad = d.rejected ?? [];
    message.value = `${d.ids?.length ?? files.length}장 추가 · 기준: ${fixed.value?.name ?? ""}`;
    if (d.skipped?.length) message.value += ` · 이미 불러온 사진 ${d.skipped.length}장 건너뜀`;
    if (bad.length)
      error.value = `제외된 파일: ${bad.map((p: any) => (typeof p === "string" ? p : (p.name ?? p.filename))).join(", ")}`;
  } catch (e) {
    report(e);
  } finally {
    uploading.value = false;
    if (fileInput.value) fileInput.value.value = "";
  }
}
async function changeFixed(id: string) {
  if (mutating.value || id === fixedId.value || tool.value === "edit") return;
  mutating.value = true;
  navigation++;
  try {
    await api("/api/fixed", { image_id: id, base_revision: revision.value });
    await refresh();
    message.value = running.value
      ? "현재 사진 처리를 마친 뒤 남은 작업을 중지하고 기준을 바꿉니다."
      : `기준 변경: ${fixed.value?.name}`;
  } catch (e) {
    report(e);
  } finally {
    mutating.value = false;
  }
}
async function remove(id: string) {
  if (running.value || tool.value === "edit") return;
  if (!confirm("이 사진을 작업 목록에서 제거할까요? 원본 파일은 유지됩니다."))
    return;
  try {
    await api(`/api/image/${id}/delete`, {});
    await refresh();
  } catch (e) {
    report(e);
  }
}
async function clearAll() {
  if (
    running.value ||
    tool.value === "edit" ||
    !confirm("현재 작업 목록을 비울까요? 원본 파일은 유지됩니다.")
  )
    return;
  try {
    await api("/api/reset", {});
    checked.value.clear();
    activeId.value = null;
    localJob.value = null;
    completedJob.value = null;
    reviewQueue = [];
    await refresh();
  } catch (e) {
    report(e);
  }
}
let historyQueue: Promise<void> = Promise.resolve();
async function undo(redo = false) {
  if (running.value || tool.value === "edit") return;
  await workspace.value?.settleMaskAction();
  if (tool.value === "adjust" && workspace.value?.draftUndo(redo)) return;
  if (!redo && workspace.value?.cancelInput()) return;
  historyQueue = historyQueue.then(() => performUndo(redo));
  return historyQueue;
}
async function performUndo(redo: boolean) {
  mutating.value = true;
  try {
    const d = await api(`/api/history/${redo ? "redo" : "undo"}`, {});
    await refresh();
    if (d.image_id && photos.value.some((p) => p.id === d.image_id))
      select(d.image_id);
    message.value = `${redo ? "다시 실행" : "되돌리기"}: ${d.label ?? ""}`;
  } catch (e) {
    report(e);
  } finally {
    mutating.value = false;
  }
}
async function run(ids: string[], recovery?: {fixedId: string; revision: number}) {
  if (!canRun.value || !ids.length) return;
  const targets = [...ids],
    jobFixedId = fixedId.value!,
    nav = navigation,
    preferred = preferredResult(targets, activeId.value);
  starting.value = true;
  error.value = "";
  message.value = `${targets.length}장 정합 준비 중…`;
  try {
    const d = await api("/api/register", {
      only: targets,
      profile: 'normal',
      lazy: lazy.value,
      anchor_only: !!recovery,
      expected_fixed: recovery?.fixedId,
      expected_anchor_revision: recovery?.revision,
    });
    localJob.value = {
      id: d.job_id,
      fixedId: jobFixedId,
      preferred,
      navigation: nav,
      targets: d.target_ids ?? targets,
    };
    running.value = true;
    if (doneDuringStart) {
      const done = doneDuringStart;
      doneDuringStart = null;
      await handleDone(done);
    }
  } catch (e) {
    report(e);
  } finally {
    starting.value = false;
  }
}
async function handleDone(d: any) {
  if (starting.value && !localJob.value) {
    doneDuringStart = d;
    return;
  }
  await refresh();
  void refreshGpu();
  const finished = localJob.value;
  if (!finished || finished.id !== d.job_id) return;
  const next = completionSelection(
    finished,
    d.job_id,
    navigation,
    activeId.value,
  );
  completedJob.value = {
    ...finished,
    resultVersions: Object.fromEntries(
      photos.value
        .filter(
          (p) =>
            finished.targets.includes(p.id) &&
            p.result?.fixed_id === finished.fixedId &&
            p.result?.job_id === finished.id &&
            !p.result?.latest_attempt_failed &&
            p.result?.status !== "fail",
        )
        .map((p) => [p.id, p.result!.id]),
    ),
  };
  localJob.value = null;
  if (next && photos.value.some((p) => p.id === next)) {
    activeId.value = next;
    if (navigation === finished.navigation) tool.value = "compare";
  }
  message.value = d.cancelled
    ? "현재 사진까지 처리하고 남은 작업을 중지했습니다."
    : `요청한 ${finished.targets.length}장 처리 완료. 사진별 결과를 확인하세요.`;
}
async function stop() {
  try {
    await api("/api/register/stop", {});
    message.value = "현재 사진을 마친 뒤 중지합니다.";
    await refresh();
  } catch (e) {
    report(e);
  }
}
async function save(ids: string[]) {
  if (!ids.length || saving.value) return;
  const expected_fixed_id = fixedId.value;
  const expected_results = Object.fromEntries(
    ids.map((id) => [id, photos.value.find((p) => p.id === id)?.result?.id]),
  );
  saving.value = true;
  error.value = '';
  exportNotice.value = `${ids.length}장 저장 파일을 준비하고 있습니다…`;
  try {
    const body = {only:ids, expected_fixed_id, expected_results};
    if (exportMode.value === 'folder') {
      if (!saveDir.value.trim()) throw Error('저장할 폴더 경로를 입력하거나 폴더 찾기를 눌러 주세요.');
      const d = await api('/api/save_results', {...body, dir:saveDir.value.trim()});
      exportNotice.value = `${d.saved}장 저장 완료 · ${d.dir}`;
      if (d.failed?.length) throw Error(`${d.saved}장 저장 · ${d.failed.length}장 실패: ${d.failed.join(', ')}`);
    } else {
      const response = await fetch('/api/export', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
      if (!response.ok) {
        const d = await response.json().catch(() => ({}));
        throw Error(d.detail || '파일을 준비하지 못했습니다. 다시 시도하세요.');
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      const filename = response.headers.get('Content-Disposition')?.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
      link.download = filename ? decodeURIComponent(filename) : (ids.length === 1 ? 'DKP-정합결과.jpg' : 'DKP-정합결과.zip');
      document.body.appendChild(link); link.click(); link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 60000);
      exportNotice.value = '다운로드를 요청했습니다. 브라우저 다운로드 목록에서 파일을 확인하세요.';
    }
    message.value = exportNotice.value;
  } catch (e) {
    exportNotice.value = e instanceof Error ? e.message : '저장하지 못했습니다.';
    report(e);
  } finally {
    saving.value = false;
  }
}
async function chooseSaveFolder() {
  if (selectingFolder.value) return;
  selectingFolder.value = true;
  exportNotice.value = '폴더 선택창을 열고 있습니다. 창이 보이지 않으면 경로를 직접 입력하거나 파일 다운로드를 선택하세요.';
  try {
    const selected = await api('/api/select_folder', {});
    if (selected.path) {saveDir.value = selected.path; exportNotice.value = `저장 폴더: ${selected.path}`;}
    else exportNotice.value = '폴더 선택을 취소했습니다.';
  } catch (e:any) {exportNotice.value = e.message;}
  finally {selectingFolder.value = false;}
}
async function review(status: string, advance = false) {
  if (!reviewable.value || !active.value) return;
  if (!reviewQueue.length)
    reviewQueue = filtered.value
      .filter((p) => p.id !== fixedId.value)
      .map((p) => p.id);
  const next = nextReview(
    reviewQueue,
    active.value.id,
    new Set(moving.value.map((p) => p.id)),
  );
  const nav = navigation,
    target = active.value.id;
  try {
    await api(`/api/result/${target}/review`, {
      result_id: result.value!.id,
      status,
    });
    await refresh();
    if (advance && next && navigation === nav && activeId.value === target) {
      activeId.value = next;
      navigation++;
      focusCanvas();
    }
    message.value =
      advance && !next
        ? "검토 목록의 마지막 사진입니다."
        : status === "confirmed"
          ? "확인됨으로 표시했습니다."
          : "보정 필요로 표시했습니다.";
  } catch (e) {
    report(e);
  }
}
function navigate(delta: number) {
  const list = moving.value,
    i = list.findIndex((p) => p.id === activeId.value),
    p = list[i + delta];
  if (p) select(p.id);
}
function keys(e: KeyboardEvent) {
  if (tool.value === "edit") return;
  const editing =
    (e.target as HTMLElement)?.closest(
      'input,textarea,select,[contenteditable="true"]',
    ) !== null;
  if (!editing && !e.isComposing && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey && e.code === 'KeyF') {
    e.preventDefault();
    toggleFocus();
    return;
  }
  const action = shortcut(e, tool.value, editing);
  if (!action) return;
  e.preventDefault();
  if (action === "undo" || action === "redo") undo(action === "redo");
  else if (action === "anchor") workspace.value?.startAnchor();
  else if (action === "delete-anchor") workspace.value?.deleteAnchor();
  else if (action === "cancel") {
    if (focused.value) focused.value = false;
    else workspace.value?.cancel();
  }
  else if (action === "confirm" || action === "reset-mask")
    workspace.value?.maskAction(action === "confirm" ? "confirm" : "reset");
  else if (action === "fit") workspace.value?.fit();
  else navigate(action === "next" ? 1 : -1);
}
let beforeEdit: Tool = "compare";
function editFixed() {
  if (!fixed.value || running.value || tool.value === "edit") return;
  navigation++;
  beforeEdit = tool.value;
  tool.value = "edit";
}
async function closeEdit(applied = false) {
  tool.value = beforeEdit;
  if (applied) {
    await refresh();
    message.value =
      "기준 편집 적용 완료. 이전 결과는 계산 당시 기준으로 보존됩니다.";
  }
  focusCanvas();
}
function badge(p: Photo) {
  if (p.id === fixedId.value) return "기준";
  if (p.result?.latest_attempt_failed) return "이번 실패 · 이전 결과";
  if (!p.result) return "미정합";
  if (p.result.status === "fail") return "정합 실패";
  if (p.result.different_reference) return "이전 기준 결과";
  if (p.result.freshness !== "current") return "다시 확인 필요";
  if (p.result.review_status === "confirmed") return "확인됨";
  if (p.result.review_status === "needs_work") return "보정 필요";
  return p.result.status === "warn" ? "검토 필요" : "정합 완료";
}
async function refreshGpu() {
  try {
    gpu.value = await api("/api/gpu");
  } catch {}
}
const restartingGpu = ref(false);
async function restartGpu() {
  if (!confirm('GPU 가속을 적용하기 위해 앱을 다시 시작합니다. 현재 사진과 결과는 초기화되므로 필요한 결과를 먼저 저장하세요. 계속할까요?')) return;
  const previous = gpu.value?.process_id;
  restartingGpu.value = true;
  try {
    await api('/api/gpu/restart', {});
    for (let i = 0; i < 120; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      try {
        const response = await fetch('/api/gpu', {signal: AbortSignal.timeout(3000)});
        const status = await response.json();
        if (response.ok && status.process_id !== previous) { location.reload(); return; }
      } catch {}
    }
    throw new Error('서버 재연결이 지연되고 있습니다. 잠시 후 화면을 새로고침하세요.');
  } catch (e) { report(e); restartingGpu.value = false; }
}
async function switchDevice(mode: 'auto' | 'cpu') {
  if (switchingDevice.value) return;
  switchingDevice.value = true;
  try {
    gpu.value = await api('/api/gpu/device', {mode});
    message.value = `${mode === 'cpu' ? 'CPU로' : 'GPU 가속으로'} 전환했습니다. 다음 계산부터 적용됩니다.`;
  } catch (e) {report(e);}
  finally {switchingDevice.value = false;}
}
async function installGpu() {
  if (
    !confirm(
      "GPU 가속에 필요한 파일을 다운로드할까요? 약 2.5GB이며 설치 후 앱을 다시 시작해야 합니다.",
    )
  )
    return;
  try {
    await api("/api/gpu/install", {});
    await refreshGpu();
  } catch (e) {
    report(e);
  }
}
onMounted(() => {
  refresh();
  refreshGpu();
  window.addEventListener("focus", refresh);
  eventSource = new EventSource("/api/events");
  eventSource.onerror = () => { connectionLost.value = true; };
  eventSource.onopen = () => {
    if (connectionLost.value) refresh();
    connectionLost.value = false;
  };
  eventSource.addEventListener("register", (event) => {
    const d = JSON.parse((event as MessageEvent).data);
    if (d.state === "done") handleDone(d);
    else if (d.state === "one_done") refresh();
    else if (d.state === "progress") {
      if (!localJob.value || d.job_id === localJob.value.id)
        message.value = `정합 ${d.done + 1}/${d.total} · ${d.name}`;
      refresh();
    } else if (d.state === "error") {
      report(d.detail);
      refresh();
    }
  });
  eventSource.addEventListener("gpu", refreshGpu);
});
onUnmounted(() => {
  eventSource?.close();
  window.removeEventListener("focus", refresh);
});
</script>

<template>
  <div
    class="app"
    @keydown.capture="exitFocus"
    :class="{ 'list-side': listSide, 'inspector-hidden': !inspector, 'focus-mode': focused, 'editing-photo': tool === 'edit' }"
    @dragover.prevent
    @drop.prevent="dropFiles"
    @dragstart.capture="guardNativeDrag"
  >
    <header class="app-header">
      <div class="brand">
        <b>DKP</b><span>Registrator <small data-testid="app-version">{{ appVersion }} · UI 12</small></span>
      </div>
      <button
        class="primary"
        :disabled="running || uploading"
        @click="fileInput?.click()"
      >
        {{ uploading ? "불러오는 중…" : "+ 사진 추가" }}
      </button>
      <button :disabled="!photos.length || running || uploading || mutating || tool === 'edit'" @click="clearAll">
        목록 비우기
      </button>
      <input
        ref="fileInput"
        type="file"
        accept="image/jpeg,image/png"
        multiple
        hidden
        aria-label="정합할 사진 추가"
        @change="upload(($event.target as HTMLInputElement).files)"
      />
      <div v-if="fixed" class="reference-summary fixed-slot" data-testid="fixed-slot"
        :class="{ 'drop-active': fixedDropActive }"
        @dragover.stop.prevent="fixedDropActive = true"
        @dragleave="fixedDropActive = false"
        @drop.stop.prevent="dropFixed"
        title="목록의 사진을 이 슬롯에 끌어 놓아 기준 사진을 교체하세요">
        <img :src="`/api/image/${fixed.id}?v=${fixed.revision}&max_side=256`" decoding="async" draggable="false" alt="" />
        <div class="fixed-slot-copy"><span class="eyebrow">고정 사진 · 끌어 놓아 교체</span>
        <strong :title="fixed.name">{{ fixed.name }}</strong></div
        ><button :disabled="running" @click="editFixed">기준 편집</button>
      </div>
      <span class="spacer" />
      <button
        :disabled="
          !history.undo_label || running || mutating || tool === 'edit'
        "
        :title="`되돌리기: ${history.undo_label || '없음'}`"
        @click="undo()"
      >
        ↶ 되돌리기
      </button>
      <button
        :disabled="
          !history.redo_label || running || mutating || tool === 'edit'
        "
        :title="`다시 실행: ${history.redo_label || '없음'}`"
        @click="undo(true)"
      >
        ↷
      </button>
      <button :aria-pressed="focused" title="사진 작업 영역 확대 · F / Esc 복귀" @click="toggleFocus">
        {{ focused ? '집중 보기 종료' : '집중 보기' }} <kbd>F</kbd>
      </button>
      <button :class="{ on: inspector && !focused }" @click="focused ? (focused = false, inspector = true) : (inspector = !inspector)">
        정보·저장
      </button>
    </header>
    <div class="main-shell">
      <main ref="surface" class="work-surface" tabindex="0" @keydown="keys">
        <template v-if="fixed && active">
          <div class="workspace-heading">
            <div>
              <span class="eyebrow">{{
                active.id === fixedId ? "기준 사진" : "현재 사진"
              }}</span
              ><strong data-testid="active-name">{{ active.name }}</strong
              ><span class="status-tag">{{ badge(active) }}</span>
            </div>
            <div class="navigation">
              <button aria-label="이전 사진" @click="navigate(-1)">←</button
              ><span
                >{{
                  Math.max(1, moving.findIndex((p) => p.id === activeId) + 1)
                }}
                / {{ moving.length || 1 }}</span
              ><button aria-label="다음 사진" @click="navigate(1)">→</button>
            </div>
          </div>
          <div v-show="tool !== 'edit'" class="tool-row">
            <div class="segmented" aria-label="작업 도구">
              <button
                v-for="t in toolNames"
                :key="t.key"
                :class="{ on: tool === t.key }"
                :disabled="
                  t.key === 'adjust' && (!result || result.status === 'fail')
                "
                @click="chooseTool(t.key)"
              >
                {{ t.name }}
              </button>
            </div>
            <span class="tool-divider" /><span class="eyebrow">보기</span>
            <div class="view-modes">
              <button
                v-for="m in modeNames"
                :key="m.key"
                :class="{ on: mode === m.key }"
                @click="chooseMode(m.key)"
              >
                {{ m.name }}
              </button>
            </div>
          </div>
          <FixedEditor
            v-if="tool === 'edit'"
            :image="fixed"
            @applied="closeEdit(true)"
            @cancel="closeEdit()"
            @error="report"
          />
          <Workspace
            v-show="tool !== 'edit'"
            ref="workspace"
            :fixed="fixed"
            :current="active"
            :tool="tool"
            :mode="mode"
            :running="running || starting || mutating"
            :revision="revision"
            :registration-error="error"
            :recovery-available="recoveryAvailable"
            @changed="refresh"
            @error="report"
            @update:tool="chooseTool"
            @recovery-busy="recoveryBusy = $event"
            @recover="({mid, fixedId, revision}) => run([mid], {fixedId, revision})"
          />
        </template>
        <section v-else class="empty-state">
          <div class="empty-mark">▧</div>
          <p class="eyebrow">사진을 고르고, 맞추고, 비교하세요</p>
          <h1>사진 비교를 한 작업대에서.</h1>
          <p>
            사진 여러 장을 추가하면 첫 사진이 기준이 됩니다.<br />기준은 언제든
            바꿀 수 있고, 마스크 없이 바로 정합할 수 있습니다.
          </p>
          <button class="primary" @click="fileInput?.click()">사진 추가</button
          ><small>JPEG · PNG · 이곳에 파일을 끌어 놓아도 됩니다</small>
        </section>
      </main>
      <aside v-if="inspector" v-show="!focused" class="inspector">
        <div class="inspector-title">
          <strong>작업 정보</strong
          ><button aria-label="정보 패널 닫기" @click="inspector = false">
            ×
          </button>
        </div>
        <template v-if="active"
          ><p class="eyebrow">현재 사진</p>
          <h2>{{ active.name }}</h2>
          <p class="subtle">
            {{ active.full_w }} × {{ active.full_h }} · {{ active.n_objects }}개
            마스크
          </p>
          <button
            class="wide"
            :disabled="!currentIsMoving || mutating || tool === 'edit'"
            @click="changeFixed(active.id)"
          >
            {{
              running
                ? "남은 작업 중지 후 기준으로 지정"
                : "이 사진을 기준으로 지정"
            }}
          </button>
          <section class="inspector-section">
            <h3>검토</h3>
            <p class="subtle">
              자동정합 완료와 사용자의 확인은 별도로 기록됩니다.
            </p>
            <button
              class="wide primary"
              :disabled="!reviewable"
              @click="review('confirmed', true)"
            >
              확인하고 다음 →</button
            ><button
              class="wide"
              :disabled="!reviewable"
              @click="review('needs_work')"
            >
              보정 필요로 표시
            </button>
          </section>
          <details v-if="result">
            <summary>정합 상세</summary>
            <dl>
              <dt>평가</dt>
              <dd>{{ result.status }}</dd>
              <dt>자동 대응점</dt>
              <dd>{{ result.n_inlier ?? "—" }}</dd>
              <dt>중앙 오차</dt>
              <dd>{{ result.reproj_median?.toFixed(2) ?? "—" }} px</dd>
              <dt>변환</dt>
              <dd>회전·이동·균일 배율</dd>
              <template v-if="result.reference_groups?.length">
                <dt>앵커 그룹별 최대 오차</dt>
                <dd v-for="(group, index) in result.reference_groups" :key="group.group">
                  기준 {{ index + 1 }} · {{ group.max_error.toFixed(1) }} px
                </dd>
              </template>
            </dl>
            <small>자동 지표는 검토를 돕는 정보입니다.</small>
          </details>
        </template>
        <section class="inspector-section">
          <h3>내보내기</h3>
          <label class="wide">저장 방식<select v-model="exportMode" :disabled="saving" aria-label="저장 방식">
            <option value="download">파일 다운로드</option><option value="folder">폴더에 저장</option>
          </select></label>
          <p v-if="exportMode === 'download'" class="subtle">한 장은 JPG, 여러 장은 ZIP으로 다운로드합니다.</p>
          <div v-else class="export-folder">
            <input v-model="saveDir" aria-label="저장 폴더 경로" placeholder="저장할 폴더 경로" :disabled="saving" />
            <button :disabled="selectingFolder || saving" @click="chooseSaveFolder">{{ selectingFolder ? '폴더 선택 대기 중…' : '폴더 찾기' }}</button>
          </div>
          <p v-if="exportNotice" class="export-notice" role="status" aria-live="polite">{{ exportNotice }}</p>
          <button
            class="wide"
            :disabled="saving || !active || !resultIds.includes(active.id)"
            @click="save([active!.id])"
          >
            현재 결과 저장</button
          ><button
            class="wide"
            :disabled="saving || !checkedMoving.some((id) => resultIds.includes(id))"
            @click="save(checkedMoving.filter((id) => resultIds.includes(id)))"
          >
            선택 결과 저장 ({{
              checkedMoving.filter((id) => resultIds.includes(id)).length
            }})</button
          ><button
            class="wide"
            :disabled="saving || !confirmedIds.length"
            @click="save(confirmedIds)"
          >
            확인한 결과 저장 ({{ confirmedIds.length }})</button
          ><button
            class="wide"
            :disabled="saving || !jobSaveIds.length"
            title="이 작업의 기준과 결과 버전이 유지된 사진만 저장합니다."
            @click="save(jobSaveIds)"
          >
            이번 작업 결과 저장</button
          ><button
            class="wide"
            :disabled="saving || !resultIds.length"
            @click="save(resultIds)"
          >
            전체 결과 저장
          </button>
        </section>
        <details @toggle="($event.target as HTMLDetailsElement).open && refreshGpu()">
          <summary>정합 설정</summary>
          <label
            ><input type="checkbox" v-model="lazy" />사진 방향 자동탐색</label
          >
          <p class="subtle">
            회전과 반전을 탐색합니다. 모든 결과의 가로세로 비율은 유지됩니다.
          </p>
          <details v-if="gpu" class="device-disclosure">
            <summary title="실행 장치 전환 펼치기/접기">
              <span>{{
              gpu.device === "cuda"
                ? "NVIDIA GPU 사용"
                : gpu.device === "mps"
                  ? "Apple Metal 사용"
                  : "CPU 사용"
              }}</span><span class="device-chevron" aria-hidden="true">⌄</span>
            </summary>
          <div v-if="gpu?.mode" class="device-test">
            <strong>실행 장치 전환 <small>(테스트용)</small></strong>
            <div>
              <button :aria-pressed="gpu.mode === 'auto' && gpu.device !== 'cpu'" :disabled="switchingDevice || running || starting || recoveryBusy || gpu.installing || !gpu.accelerator" @click="switchDevice('auto')">GPU 가속</button>
              <button :aria-pressed="gpu.device === 'cpu'" :disabled="switchingDevice || running || starting || recoveryBusy || gpu.installing" @click="switchDevice('cpu')">CPU</button>
            </div>
            <p class="subtle">{{ switchingDevice ? '장치를 전환하고 있습니다…' : '다음 계산부터 적용됩니다. 전환 후 첫 작업은 모델 준비로 더 걸릴 수 있으며, 앱 재실행 시 자동 선택으로 돌아갑니다.' }}</p>
            <p v-for="(device, name) in gpu.models" :key="name" class="subtle">{{ name }} 모델: {{ device }}</p>
          </div>
          </details>
          <p v-if="gpu?.device === 'cpu'" class="subtle" data-testid="cpu-help">{{ cpuHelp }}</p>
          <details v-if="gpu?.platform === 'win32'" class="gpu-requirements">
            <summary title="NVIDIA GPU와 NVIDIA 드라이버가 필요합니다. CUDA Toolkit을 별도로 설치할 필요는 없습니다.">GPU 사용 조건 ⓘ</summary>
            <p>{{ gpu.gpu_name ? `✓ 감지된 GPU: ${gpu.gpu_name}` : 'NVIDIA GPU를 감지하지 못했습니다. NVIDIA 그래픽카드가 있는지 확인하고 NVIDIA 드라이버를 설치·업데이트한 뒤 앱을 다시 시작하세요.' }}</p>
            <p>{{ gpu.frozen ? '✓ Windows 실행 파일 사용 중' : '앱 내 자동 설치는 Windows 배포용 실행 파일에서 지원합니다. 소스 실행은 README의 GPU 환경 설정을 이용하세요.' }}</p>
            <p>CUDA는 NVIDIA GPU에서 사용합니다. Intel·AMD 그래픽만 있는 PC는 CPU 모드를 사용하세요.</p>
            <p>인터넷 연결과 다운로드·압축 해제에 충분한 디스크 여유 공간이 필요합니다. CUDA Toolkit은 별도로 설치하지 않아도 됩니다.</p>
            <p>GPU가 감지돼도 드라이버와 설치된 PyTorch CUDA의 호환 여부에 따라 가속을 사용할 수 없을 수 있습니다.</p>
          </details>
          <button
            v-if="
              gpu?.gpu_name &&
              gpu.frozen &&
              gpu.device === 'cpu' &&
              !gpu.installed
            "
            :disabled="gpu.installing"
            @click="installGpu"
          >
            {{ gpu.installing ? "GPU 설치 중…" : "GPU 가속 설치" }}
          </button>
          <p v-if="gpu?.error" class="failure">{{ gpu.error }}</p>
          <GpuInstallProgress :gpu="gpu" />
          <button v-if="gpu?.can_restart && (gpu.installed || gpu.phase === 'done') && gpu.device === 'cpu'" :disabled="restartingGpu || gpu.installing || running || starting" @click="restartGpu">{{ restartingGpu ? '다시 시작 중…' : 'GPU 적용하고 다시 시작' }}</button>
        </details>
        <details>
          <summary>단축키</summary>
          <p>
            Z 마스크 확정 · X 마스크 초기화<br />A 대응점 추가 · D 선택점
            취소<br />Ctrl/Cmd+Z 되돌리기<br />Ctrl/Cmd+Shift+Z 다시 실행<br />0
            화면 맞춤 · Space+드래그 이동<br />F 집중 보기 · Esc 복귀
          </p>
          <small
            >작업대에 포커스가 있을 때 사용합니다. C는 지정하지
            않았습니다.</small
          >
        </details>
      </aside>
      <section v-show="!focused" class="photo-browser" aria-label="사진 목록">
        <div class="browser-toolbar">
          <strong>사진 {{ photos.length }}</strong
          ><input
            type="search"
            placeholder="파일명 검색"
            aria-label="파일명 검색"
            v-model="search"
          /><select v-model="filter" aria-label="사진 필터">
            <option value="all">전체</option>
            <option value="unregistered">미정합</option>
            <option value="review">검토 필요</option>
            <option value="failed">실패</option>
            <option value="confirmed">확인됨</option></select
          ><button
            @click="
              checked = new Set(
                filtered.filter((p) => p.id !== fixedId).map((p) => p.id),
              )
            "
          >
            전체 선택</button
          ><button @click="checked = new Set()">선택 해제</button
          ><button
            @click="
              checked = new Set(
                moving
                  .filter(
                    (p) =>
                      p.result?.status === 'fail' ||
                      p.result?.latest_attempt_failed,
                  )
                  .map((p) => p.id),
              )
            "
          >
            실패 선택</button
          ><span class="spacer" /><button
            :aria-pressed="listSide"
            @click="listSide = !listSide"
          >
            {{ listSide ? "아래 필름스트립" : "왼쪽 목록" }}
          </button>
        </div>
        <div class="filmstrip">
          <article
            v-for="(p, i) in filtered"
            :key="p.id"
            class="photo-card"
            :class="{
              active: p.id === activeId,
              reference: p.id === fixedId,
              checked: checked.has(p.id),
              dragging: draggingPhoto === p.id,
              'insert-before': insertion?.id === p.id && !insertion.after,
              'insert-after': insertion?.id === p.id && insertion.after,
            }"
            :data-photo-id="p.id"
            :draggable="tool !== 'edit'"
            @dragstart="dragPhoto($event, p.id)"
            @dragend="endPhotoDrag"
            @dragover.stop="previewOrder($event, p.id)"
            @dragleave="insertion?.id === p.id && (insertion = null)"
            @drop.stop.prevent="dropPhoto($event, p.id)"
            @click="select(p.id)"
          >
            <button
              class="thumbnail-button"
              :aria-label="`사진 보기 ${p.name}`"
              title="끌어서 순서 변경 · 상단 고정 슬롯에 놓으면 기준 변경"
            >
              <img
                loading="lazy"
                decoding="async"
                draggable="false"
                :src="`/api/image/${p.id}?v=${p.revision}&max_side=256`"
                :alt="p.name"
              /><span class="photo-index">{{ i + 1 }}</span
              ><span v-if="p.id === fixedId" class="reference-badge"
                >기준</span
              ></button
            ><input
              v-if="p.id !== fixedId"
              type="checkbox"
              :aria-label="`일괄 선택 ${p.name}`"
              :checked="checked.has(p.id)"
              @click.stop
              @change="toggle(p.id)"
            />
            <div class="card-copy">
              <strong :title="p.name">{{ p.name }}</strong
              ><span>{{
                job?.items?.[p.id] === "running"
                  ? "처리 중…"
                  : job?.items?.[p.id] === "queued"
                    ? "대기 중"
                    : badge(p)
              }}</span>
            </div>
            <button
              class="remove-photo"
              :disabled="running || tool === 'edit'"
              :aria-label="`목록에서 제거 ${p.name}`"
              @click.stop="remove(p.id)"
            >
              ×
            </button>
          </article>
          <p v-if="photos.length && !filtered.length" class="subtle">
            조건에 맞는 사진이 없습니다. 필터를 바꾸면 현재 사진은 그대로
            유지됩니다.
          </p>
        </div>
      </section>
    </div>
    <footer class="action-bar">
      <span class="selection-summary"
        >현재 {{ currentIsMoving ? "1" : "0" }}장 · 체크
        {{ checkedMoving.length }}장</span
      ><button
        class="primary"
        :disabled="!canRun || !currentIsMoving"
        @click="run([activeId!])"
      >
        현재 정합</button
      ><button
        :disabled="!canRun || !checkedMoving.length"
        @click="run(checkedMoving)"
      >
        선택 {{ checkedMoving.length }}장 정합</button
      ><button :disabled="!canRun" @click="run(moving.map((p) => p.id))">
        전체 정합</button
      ><button v-if="running" class="danger" @click="stop">
        남은 작업 중지</button
      ><span class="spacer" /><button
        v-if="completedJob"
        @click="
          select(completedJob.preferred!);
          tool = 'compare';
        "
      >
        최근 결과 보기</button>
    </footer>
    <div class="status-bar" role="status" aria-live="polite">
      <span v-if="connectionLost" class="connection-warning">서버 연결이 끊겼습니다 · 자동 재연결 중…</span>
      <span v-else :class="{ failure: error }">{{ error || message }}</span
      ><button v-if="error" aria-label="오류 메시지 닫기" @click="error = ''">
        ×</button
      ><span class="spacer" /><span class="privacy-note">로컬 사진 처리</span>
    </div>
  </div>
</template>
