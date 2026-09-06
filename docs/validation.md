# Workspace validation

Validated on Windows with Python 3.13, Node and Chromium on 2026-09-06.

- Python API and engine: 30 tests covering reference changes, revision/history, masks, anchors, exports, failure isolation, immutable results and similarity geometry.
- Frontend: 10 unit tests, type checking and production build.
- Browser workflows cover upload order, fifth-image completion, navigation races, Z/X/A/D and undo, review queue, original-resolution ROI, integrated editor, manual adjustment, job-bound export and responsive layouts (1366×768, 1920×1080, effective 125%/150%).
- Native release workflows start the actual packaged executable and import a synthetic image before publishing it.

## Thirty-photo navigation measurement

One local isolated Chromium run used 30 synthetic 1600×1000 PNGs: upload/list ready 1386 ms; five photo selections 52, 39, 33, 34 and 54 ms. Chromium JavaScript heap was 6.1 MiB. This excludes decoded images, GPU and Python native memory and is not a total application memory measurement. Timings vary by machine, image size and cache. It is not a clinical registration benchmark.

The browser suite uses deterministic synthetic masking/registration to check interaction and state transitions; real engine geometry has separate Python tests. Clinical accuracy on representative patient photographs requires separate evaluation. No patient photographs were uploaded for these checks.

## Session startup correction

The previous server erased its shared temporary session directory on import. The initial test collection encountered that old behavior before isolation was installed. Original files outside that temporary directory were unaffected; whether earlier temporary work existed was not recorded. Startup now creates a unique session directory without deleting existing sessions, and all tests use isolated temporary roots. Restart recovery remains a future feature.
