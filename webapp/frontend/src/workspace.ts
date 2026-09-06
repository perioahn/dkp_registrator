export type Tool = "compare" | "mask" | "anchor" | "adjust" | "edit";
export type ComparisonMode = "side" | "wipe" | "false" | "flicker" | "match";
export type Point = [number, number];
export interface Viewport {
  zoom: number;
  cx: number;
  cy: number;
}
export interface ResultInfo {
  job_id?: string;
  id: string;
  status: string;
  gate: string;
  label: string;
  reason?: string;
  fixed_id: string;
  fixed_name?: string;
  different_reference?: boolean;
  fixed_revision: number | string;
  freshness: string;
  review_status: string;
  latest_attempt_failed?: boolean;
  manual_adjusted?: boolean;
  used_mask?: boolean;
  has_previous?: boolean;
  previous?: {
    id: string;
    full_w: number;
    full_h: number;
    fixed_id: string;
    fixed_name?: string;
    fixed_revision: number | string;
  };
  n_inlier?: number;
  inlier_ratio?: number;
  reproj_median?: number;
  anchor_residual?: number;
  full_w?: number;
  full_h?: number;
  width?: number;
  height?: number;
}
export interface Photo {
  id: string;
  name: string;
  role: "fixed" | "moving";
  w: number;
  h: number;
  full_w: number;
  full_h: number;
  source_w: number;
  source_h: number;
  revision: number | string;
  G: number[][];
  edits?: any;
  n_objects: number;
  mask_ready: boolean;
  mask_rev: number;
  result: ResultInfo | null;
}
export interface Anchor {
  id: string;
  fixed: Point;
  moving: Point;
  enabled: boolean;
  requested_enabled?: boolean;
}
export interface LocalJob {
  id: string;
  fixedId: string;
  resultVersions?: Record<string, string>;
  preferred: string | null;
  navigation: number;
  targets: string[];
}
export const identity = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];
export function mapPoint(G: number[][] | undefined, p: Point): Point {
  const m = G ?? identity;
  return [
    m[0][0] * p[0] + m[0][1] * p[1] + m[0][2],
    m[1][0] * p[0] + m[1][1] * p[1] + m[1][2],
  ];
}
export function inversePoint(G: number[][] | undefined, p: Point): Point {
  const m = G ?? identity,
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12)
    throw Error("유효하지 않은 사진 좌표입니다");
  const x = p[0] - m[0][2],
    y = p[1] - m[1][2];
  return [
    (m[1][1] * x - m[0][1] * y) / det,
    (-m[1][0] * x + m[0][0] * y) / det,
  ];
}
export function preferredResult(targets: string[], active: string | null) {
  return active && targets.includes(active) ? active : (targets[0] ?? null);
}
export function completionSelection(
  job: Pick<LocalJob, "id" | "preferred" | "navigation"> | null,
  id: string,
  navigation: number,
  active: string | null,
) {
  return job?.id === id && job.navigation === navigation
    ? job.preferred
    : active;
}
export function nextReview(
  queue: string[],
  active: string,
  available: Set<string>,
) {
  return (
    queue.slice(queue.indexOf(active) + 1).find((id) => available.has(id)) ??
    null
  );
}
type Key = Pick<
  KeyboardEvent,
  | "code"
  | "ctrlKey"
  | "metaKey"
  | "altKey"
  | "shiftKey"
  | "isComposing"
  | "repeat"
>;
export function shortcut(e: Key, tool: Tool, editing: boolean): string | null {
  if (editing || e.isComposing || e.repeat) return null;
  if (e.ctrlKey || e.metaKey)
    return !e.altKey && e.code === "KeyZ"
      ? e.shiftKey
        ? "redo"
        : "undo"
      : null;
  if (e.altKey || e.shiftKey) return null;
  if (e.code === "Escape") return "cancel";
  if (e.code === "Digit0") return "fit";
  if (e.code === "ArrowLeft") return "previous";
  if (e.code === "ArrowRight") return "next";
  if (e.code === "KeyA") return "anchor";
  if (tool !== "adjust" && tool !== "edit") {
    if (e.code === "KeyD") return "delete-anchor";
    if (e.code === "KeyZ") return "confirm";
    if (e.code === "KeyX") return "reset-mask";
  }
  return null;
}
export function zoomAt(
  v: Viewport,
  newZoom: number,
  x: number,
  y: number,
  cw: number,
  ch: number,
  w: number,
  h: number,
  fit: number,
): Viewport {
  const s = fit * v.zoom,
    ns = fit * newZoom;
  return {
    zoom: newZoom,
    cx: v.cx + ((x - cw / 2) * (1 / s - 1 / ns)) / w,
    cy: v.cy + ((y - ch / 2) * (1 / s - 1 / ns)) / h,
  };
}
export async function api<T = any>(
  url: string,
  body?: unknown,
  method = "POST",
): Promise<T> {
  const res = await fetch(
    url,
    body === undefined
      ? undefined
      : {
          method,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await res.json();
  if (!res.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : `요청 실패 (${res.status})`,
    );
  return data;
}
