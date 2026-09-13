import { afterEach, expect, it, vi } from "vitest";
import { getPreview } from "./preview";
import {
  clearEditorPreviewCache,
  editorPreviewCacheSize,
  getEditorPreview,
} from "./editorPreview";

afterEach(() => {
  clearEditorPreviewCache();
  vi.unstubAllGlobals();
});

function response() {
  return new Response(new Blob(["preview"], { type: "image/jpeg" }), {
    headers: { "X-Source-Width": "6000", "X-Source-Height": "4000" },
  });
}

it("reuses a bounded server preview without rebuilding it in the browser", async () => {
  const fetcher = vi.fn(async () => response());
  const decode = vi.fn();
  vi.stubGlobal("fetch", fetcher);
  vi.stubGlobal("createImageBitmap", decode);
  const image = { id: "fixed", name: "fixed.jpg", revision: 3 };

  const first = await getEditorPreview(image);
  const second = await getEditorPreview(image);
  const asset = await getPreview(first.file);

  expect(fetcher).toHaveBeenCalledTimes(1);
  expect(second.file).toBe(first.file);
  expect(asset.preview.size).toBe(first.file.size);
  expect(asset.preview.type).toBe("image/jpeg");
  expect(asset.width).toBe(6000);
  expect(asset.height).toBe(4000);
  expect(decode).not.toHaveBeenCalled();
});

it("keys previews by revision and bounds retained photos", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => response()));
  for (let revision = 0; revision < 4; revision++)
    await getEditorPreview({ id: "fixed", name: "fixed.jpg", revision });
  expect(editorPreviewCacheSize()).toBe(3);
});
