export interface PreviewAsset {
  file: File;
  preview: Blob;
  thumbnail: Blob;
  width: number;
  height: number;
}

async function decodeSource(file: File): Promise<File> {
  if (!/\.nef$/i.test(file.name)) return file;
  const bytes = new Uint8Array(await file.arrayBuffer());
  let start = -1,
    bestStart = 0,
    bestLength = 0;
  for (let i = 0; i < bytes.length - 3; i++) {
    if (
      start < 0 &&
      bytes[i] === 255 &&
      bytes[i + 1] === 216 &&
      bytes[i + 2] === 255
    ) {
      start = i;
      i += 2;
    } else if (start >= 0 && bytes[i] === 255 && bytes[i + 1] === 217) {
      if (i + 2 - start > bestLength) {
        bestStart = start;
        bestLength = i + 2 - start;
      }
      start = -1;
      i++;
    }
  }
  if (bestLength < 100000)
    throw new Error(`${file.name}: 내장 JPEG 미리보기를 찾지 못했습니다.`);
  return new File([bytes.slice(bestStart, bestStart + bestLength)], file.name, {
    type: "image/jpeg",
  });
}

export async function buildPreview(input: File): Promise<PreviewAsset> {
  const file = await decodeSource(input);
  const bitmap = await createImageBitmap(file, {
    imageOrientation: "from-image",
  });
  try {
    if (!bitmap.width || !bitmap.height)
      throw new Error("사진 크기를 읽지 못했습니다.");
    // Legacy-browser fallback preserves functionality; modern browsers use the worker.
    if (typeof OffscreenCanvas === "undefined")
      return {
        file,
        preview: file,
        thumbnail: file,
        width: bitmap.width,
        height: bitmap.height,
      };
    const s = Math.min(1, 1600 / Math.max(bitmap.width, bitmap.height));
    const canvas = new OffscreenCanvas(
      Math.max(1, Math.round(bitmap.width * s)),
      Math.max(1, Math.round(bitmap.height * s)),
    );
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const preview = await canvas.convertToBlob({
      type: "image/jpeg",
      quality: 0.94,
    });
    const t = Math.min(1, 256 / Math.max(canvas.width, canvas.height));
    const thumb = new OffscreenCanvas(
      Math.max(1, Math.round(canvas.width * t)),
      Math.max(1, Math.round(canvas.height * t)),
    );
    thumb.getContext("2d")!.drawImage(canvas, 0, 0, thumb.width, thumb.height);
    return {
      file,
      preview,
      thumbnail: await thumb.convertToBlob({
        type: "image/jpeg",
        quality: 0.8,
      }),
      width: bitmap.width,
      height: bitmap.height,
    };
  } finally {
    bitmap.close();
  }
}
