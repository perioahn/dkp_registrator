import { buildPreview, type PreviewAsset } from "./previewCore";

const cache = new Map<File, Promise<PreviewAsset>>();
const pending = new Map<
  number,
  { resolve: (asset: PreviewAsset) => void; reject: (error: Error) => void }
>();
let worker: Worker | null = null,
  serial = 0;
function decode(file: File): Promise<PreviewAsset> {
  if (typeof Worker === "undefined" || typeof OffscreenCanvas === "undefined")
    return buildPreview(file);
  if (!worker) {
    worker = new Worker(new URL("./preview.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = ({ data }) => {
      const job = pending.get(data.id);
      pending.delete(data.id);
      if (data.error) job?.reject(new Error(data.error));
      else job?.resolve(data.asset);
    };
    worker.onerror = () => {
      for (const job of pending.values())
        job.reject(
          new Error(
            "사진 미리보기 작업을 시작하지 못했습니다. 다시 열어주세요.",
          ),
        );
      pending.clear();
      worker?.terminate();
      worker = null;
      cache.clear();
    };
  }
  return new Promise((resolve, reject) => {
    const id = ++serial;
    pending.set(id, { resolve, reject });
    worker!.postMessage({ id, file });
  });
}
export function rememberPreview(file: File, asset: Promise<PreviewAsset>) {
  cache.delete(file);
  cache.set(file, asset);
  while (cache.size > 3) cache.delete(cache.keys().next().value!);
}
export function getPreview(file: File): Promise<PreviewAsset> {
  const found = cache.get(file);
  if (found) {
    rememberPreview(file, found);
    return found;
  }
  const promise = decode(file).catch((error) => {
    cache.delete(file);
    throw error;
  });
  rememberPreview(file, promise);
  return promise;
}
export function forgetPreview(file: File) {
  cache.delete(file);
}
export function previewCacheSize() {
  return cache.size;
}
