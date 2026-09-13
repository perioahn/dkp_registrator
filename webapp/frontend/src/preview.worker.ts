import { buildPreview } from "./previewCore";
// Serial decoding bounds full-resolution memory even during rapid navigation.
let queue = Promise.resolve();
self.onmessage = (event: MessageEvent) => {
  const { id, file } = event.data;
  queue = queue.then(async () => {
    try {
      self.postMessage({ id, asset: await buildPreview(file) });
    } catch (error) {
      self.postMessage({
        id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
};
