import { rememberPreview } from "./preview";
import type { PreviewAsset } from "./previewCore";

export interface EditorPreviewSource {
  id: string;
  name: string;
  revision: number | string;
}

export interface EditorPreviewPhoto {
  id: string;
  name: string;
  file: File;
  sourceWidth: number;
  sourceHeight: number;
}

interface CachedEditorPreview {
  photo: EditorPreviewPhoto;
  asset: PreviewAsset;
}

const cache = new Map<string, Promise<CachedEditorPreview>>();
const CACHE_LIMIT = 3;

function cacheKey(image: EditorPreviewSource) {
  return `${image.id}:${image.revision}`;
}

function retain(key: string, value: Promise<CachedEditorPreview>) {
  cache.delete(key);
  cache.set(key, value);
  while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value!);
}

export function getEditorPreview(
  image: EditorPreviewSource,
): Promise<EditorPreviewPhoto> {
  const key = cacheKey(image);
  let pending = cache.get(key);
  if (!pending) {
    pending = fetch(`/api/image/${image.id}/source-preview`)
      .then(async (response) => {
        if (!response.ok) throw new Error("원본 사진을 읽지 못했습니다.");
        const blob = await response.blob();
        const sourceWidth = Number(response.headers.get("X-Source-Width"));
        const sourceHeight = Number(response.headers.get("X-Source-Height"));
        if (!sourceWidth || !sourceHeight)
          throw new Error("원본 크기 정보를 읽지 못했습니다.");
        const file = new File([blob], "preview.jpg", { type: blob.type });
        const photo = {
          id: image.id,
          name: image.name,
          file,
          sourceWidth,
          sourceHeight,
        };
        return {
          photo,
          asset: {
            file,
            preview: blob,
            thumbnail: blob,
            width: sourceWidth,
            height: sourceHeight,
          },
        };
      })
      .catch((error) => {
        cache.delete(key);
        throw error;
      });
    retain(key, pending);
  } else retain(key, pending);
  return pending.then(({ photo, asset }) => {
    // source-preview is already bounded and encoded by the server. Register it
    // directly so PhotoEditorCanvas does not decode, resize and encode it again.
    rememberPreview(photo.file, Promise.resolve(asset));
    return photo;
  });
}

export function prefetchEditorPreview(image: EditorPreviewSource) {
  void getEditorPreview(image).catch(() => {});
}

export function clearEditorPreviewCache() {
  cache.clear();
}

export function editorPreviewCacheSize() {
  return cache.size;
}
