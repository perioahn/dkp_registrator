"""DKP Registrator 웹 UI 백엔드 — FastAPI.

기존 엔진(register.py/sam2_mask.py/config.py)을 그대로 사용하는 브라우저 UI.
tkinter GUI(main_gui.py)와 병행 제공. 실행: py -3.13 webapp/server.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 엔진의 진행 로그에 유니코드(—, ° 등)가 있어 cp949 콘솔에서 크래시하지 않도록
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import PROFILES, get_profile  # noqa: E402
from register import register_test, register_test_lazy  # noqa: E402

log = logging.getLogger(__name__)

SAM2_MAX_SIDE = 1024


def data_dir() -> str:
    base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~")) \
        if os.name == "nt" else os.path.expanduser(
            "~/Library/Application Support" if sys.platform == "darwin"
            else "~/.local/share")
    d = os.path.join(base, "DKPRegistratorWeb")
    os.makedirs(d, exist_ok=True)
    return d


# ── 세션 (단일 활성) ────────────────────────────────

class Session:
    """이미지·마스크·앵커·결과를 담는 단일 작업 세션."""

    def __init__(self):
        self.dir = os.path.join(data_dir(), "session")
        shutil.rmtree(self.dir, ignore_errors=True)
        os.makedirs(self.dir)
        self.images: dict[str, dict] = {}   # id → {role, name, path}
        self.order: list[str] = []          # 업로드 순서 (fixed 먼저)
        self.masks: dict[str, dict] = {}    # id → {points:[], confirmed:[np], current:np|None}
        self.results: dict[str, dict] = {}  # moving_id → result entry
        self.running = False

    def fixed_id(self) -> str | None:
        return next((i for i in self.order
                     if self.images[i]["role"] == "fixed"), None)

    def moving_ids(self) -> list[str]:
        return [i for i in self.order if self.images[i]["role"] == "moving"]


SESSION = Session()

_img_cache: dict[str, np.ndarray] = {}   # id → RGB (SAM2_MAX_SIDE 제한)
_full_cache: dict[str, np.ndarray] = {}  # id → RGB 원본


def _load_rgb(path: str) -> np.ndarray:
    """EXIF 방향 반영 RGB 로드 (main_gui.load_image_rgb와 동일 정책)."""
    from PIL import Image, ImageOps
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return np.array(img.convert("RGB"))


def get_full(img_id: str) -> np.ndarray:
    if img_id not in _full_cache:
        _full_cache[img_id] = _load_rgb(SESSION.images[img_id]["path"])
    return _full_cache[img_id]


def get_work(img_id: str) -> np.ndarray:
    """SAM2/화면용 축소본 (최대 1024px)."""
    if img_id not in _img_cache:
        full = get_full(img_id)
        h, w = full.shape[:2]
        s = SAM2_MAX_SIDE / max(h, w)
        if s < 1:
            full = cv2.resize(full, (int(w * s), int(h * s)),
                              interpolation=cv2.INTER_AREA)
        _img_cache[img_id] = full
    return _img_cache[img_id]


def work_scale(img_id: str) -> float:
    """work → full 배율 (full = work × scale)."""
    full = get_full(img_id)
    work = get_work(img_id)
    return full.shape[0] / work.shape[0]


# ── SSE 브로커 ─────────────────────────────────────

_loop: asyncio.AbstractEventLoop | None = None
_subs: set[asyncio.Queue] = set()
_sub_lock = threading.Lock()


def publish(event: str, data: dict) -> None:
    if _loop is None:
        return
    msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    with _sub_lock:
        subs = list(_subs)
    _loop.call_soon_threadsafe(
        lambda: [q.put_nowait(msg) for q in subs if q.qsize() < 500])


# ── SAM2 ───────────────────────────────────────────

_sam = None
_sam_lock = threading.Lock()
_sam_current: str | None = None  # 현재 set_image된 이미지 id


def _get_sam():
    global _sam
    if _sam is None:
        from sam2_mask import load_sam2_predictor
        _sam = load_sam2_predictor()
    return _sam


def _sam_select(img_id: str) -> None:
    global _sam_current
    if _sam_current != img_id:
        from sam2_mask import sam_set_image
        sam_set_image(_get_sam(), get_work(img_id))
        _sam_current = img_id


def _mask_state(img_id: str) -> dict:
    return SESSION.masks.setdefault(
        img_id, {"points": [], "confirmed": [], "current": None})


def _predict_mask(img_id: str) -> None:
    from sam2_mask import sam_predict
    st = _mask_state(img_id)
    if not st["points"]:
        st["current"] = None
        return
    pts = np.array([[p["x"], p["y"]] for p in st["points"]], dtype=np.float32)
    lbl = np.array([p["label"] for p in st["points"]], dtype=np.int32)
    with _sam_lock:
        _sam_select(img_id)
        masks, scores, _ = sam_predict(_get_sam(), get_work(img_id), pts, lbl)
    st["current"] = masks[int(np.argmax(scores))].astype(bool)


def _union_mask(img_id: str) -> np.ndarray | None:
    """확정 개체 + 현재 작업분 union (work 해상도, uint8)."""
    st = _mask_state(img_id)
    parts = list(st["confirmed"])
    if st["current"] is not None:
        parts.append(st["current"])
    if not parts:
        return None
    u = np.zeros(get_work(img_id).shape[:2], dtype=bool)
    for m in parts:
        u |= m
    return (u * 255).astype(np.uint8)


def _mask_overlay_png(img_id: str) -> bytes:
    """현재(노랑) + 확정(파랑) 마스크를 RGBA PNG로."""
    st = _mask_state(img_id)
    h, w = get_work(img_id).shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for m in st["confirmed"]:
        rgba[m] = (60, 120, 255, 110)
    if st["current"] is not None:
        cur = st["current"]
        rgba[cur] = (255, 210, 40, 130)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return buf.tobytes()


# ── FastAPI ────────────────────────────────────────

app = FastAPI(title="dkp-registrator-web")


@app.middleware("http")
async def local_guard(request: Request, call_next):
    host = (request.headers.get("host") or "").split(":")[0]
    if host not in ("127.0.0.1", "localhost"):
        return JSONResponse({"detail": "bad host"}, status_code=403)
    return await call_next(request)


@app.on_event("startup")
async def startup():
    global _loop
    _loop = asyncio.get_running_loop()


@app.get("/api/state")
def state() -> dict:
    def img_info(i):
        st = SESSION.masks.get(i, {})
        work = get_work(i)
        return {
            "id": i, "role": SESSION.images[i]["role"],
            "name": SESSION.images[i]["name"],
            "w": work.shape[1], "h": work.shape[0],
            "n_objects": len(st.get("confirmed", [])),
            "has_current": st.get("current") is not None,
            "mask_ready": _union_mask(i) is not None,
            "result": _result_summary(SESSION.results.get(i)),
        }
    return {
        "images": [img_info(i) for i in SESSION.order],
        "fixed": SESSION.fixed_id(),
        "running": SESSION.running,
        "profiles": list(PROFILES),
    }


def _result_summary(r: dict | None) -> dict | None:
    if not r:
        return None
    m = r.get("metrics") or {}
    return {
        "status": r.get("status"), "gate": r.get("gate"),
        "label": r.get("label"), "reason": r.get("reason"),
        "n_inlier": m.get("n_inlier"), "inlier_ratio": m.get("inlier_ratio"),
        "reproj_median": m.get("reproj_median"),
        "rotation_deg": m.get("rotation_deg"), "scale": m.get("scale"),
    }


@app.post("/api/reset")
def reset() -> dict:
    global SESSION, _img_cache, _full_cache, _sam_current
    if SESSION.running:
        raise HTTPException(409, "정합 실행 중")
    SESSION = Session()
    _img_cache = {}
    _full_cache = {}
    _sam_current = None
    return {"ok": True}


@app.post("/api/upload")
async def upload(role: str, files: list[UploadFile] = File(...)) -> dict:
    if role not in ("fixed", "moving"):
        raise HTTPException(400, "role must be fixed|moving")
    if role == "fixed" and SESSION.fixed_id():
        raise HTTPException(409, "fixed는 1장만 — 기존 것을 삭제하세요")
    added = []
    for f in files:
        img_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(f.filename or "img.jpg")[1] or ".jpg"
        path = os.path.join(SESSION.dir, img_id + ext)
        with open(path, "wb") as out:
            out.write(await f.read())
        try:
            _load_rgb(path)  # 로드 가능 검증
        except Exception:
            os.remove(path)
            raise HTTPException(400, f"이미지 로드 실패: {f.filename}")
        SESSION.images[img_id] = {"role": role, "name": f.filename, "path": path}
        SESSION.order.append(img_id)
        added.append(img_id)
        if role == "fixed":
            break  # fixed는 첫 파일만
    return {"added": added}


@app.post("/api/image/{img_id}/delete")
def delete_image(img_id: str) -> dict:
    if img_id not in SESSION.images:
        raise HTTPException(404)
    SESSION.order.remove(img_id)
    SESSION.images.pop(img_id)
    SESSION.masks.pop(img_id, None)
    SESSION.results.pop(img_id, None)
    _img_cache.pop(img_id, None)
    _full_cache.pop(img_id, None)
    return {"ok": True}


@app.get("/api/image/{img_id}")
def serve_image(img_id: str):
    if img_id not in SESSION.images:
        raise HTTPException(404)
    work = get_work(img_id)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(work, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 88])
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.get("/api/mask/{img_id}/overlay")
def mask_overlay(img_id: str):
    if img_id not in SESSION.images:
        raise HTTPException(404)
    return StreamingResponse(io.BytesIO(_mask_overlay_png(img_id)),
                             media_type="image/png")


@app.post("/api/mask/{img_id}/click")
async def mask_click(img_id: str, x: float = Body(embed=True),
                     y: float = Body(embed=True),
                     label: int = Body(embed=True)) -> dict:
    """캔버스 클릭 (label 1=포함, 0=제외) → SAM2 예측."""
    if img_id not in SESSION.images:
        raise HTTPException(404)
    st = _mask_state(img_id)
    st["points"].append({"x": x, "y": y, "label": label})
    await asyncio.to_thread(_predict_mask, img_id)
    return {"points": st["points"], "ts": time.time_ns()}


@app.post("/api/mask/{img_id}/action")
async def mask_action(img_id: str, action: str = Body(embed=True)) -> dict:
    """confirm(개체 확정=Z) / undo(마지막 점 취소) / reset(전체 초기화=X)."""
    st = _mask_state(img_id)
    if action == "confirm":
        if st["current"] is not None:
            st["confirmed"].append(st["current"])
        st["points"] = []
        st["current"] = None
    elif action == "undo":
        if st["points"]:
            st["points"].pop()
            await asyncio.to_thread(_predict_mask, img_id)
        elif st["confirmed"]:
            st["confirmed"].pop()
    elif action == "reset":
        st["points"] = []
        st["confirmed"] = []
        st["current"] = None
    else:
        raise HTTPException(400, "unknown action")
    return {"n_objects": len(st["confirmed"]), "points": st["points"],
            "ts": time.time_ns()}


# ── 정합 실행 ──────────────────────────────────────

def _run_registration(lazy: bool, profile: str) -> None:
    cfg = get_profile(profile)
    fixed_id = SESSION.fixed_id()
    movings = SESSION.moving_ids()
    try:
        f_scale = work_scale(fixed_id)
        fixed_full = get_full(fixed_id)
        fmask = _union_mask(fixed_id)
        fmask_full = cv2.resize(fmask, (fixed_full.shape[1], fixed_full.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        total = len(movings)
        for i, mid in enumerate(movings):
            publish("register", {"state": "progress", "done": i, "total": total,
                                 "name": SESSION.images[mid]["name"]})
            m_full = get_full(mid)
            mmask = _union_mask(mid)
            mmask_full = cv2.resize(mmask, (m_full.shape[1], m_full.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

            def cb(cur, tot, label, _mid=mid, _i=i):
                publish("register", {"state": "lazy", "done": _i, "total": total,
                                     "lazy_cur": cur, "lazy_total": tot,
                                     "lazy_label": label})

            fn = register_test_lazy if lazy else register_test
            kw = {"cfg": cfg}
            if lazy:
                kw["progress_callback"] = cb
            entry = fn(fixed_full, m_full, fmask_full, mmask_full, **kw)[0]
            SESSION.results[mid] = entry
            publish("register", {"state": "one_done", "id": mid,
                                 "summary": _result_summary(entry)})
        publish("register", {"state": "done", "total": total})
    except Exception as e:
        log.exception("registration failed")
        publish("register", {"state": "error", "detail": str(e)})
    finally:
        SESSION.running = False


@app.post("/api/register")
def run_register(lazy: bool = Body(default=False, embed=True),
                 profile: str = Body(default="normal", embed=True)) -> dict:
    if SESSION.running:
        raise HTTPException(409, "이미 실행 중")
    fixed_id = SESSION.fixed_id()
    if not fixed_id or _union_mask(fixed_id) is None:
        raise HTTPException(409, "Fixed 이미지와 마스크가 필요합니다")
    movings = [m for m in SESSION.moving_ids() if _union_mask(m) is not None]
    if not movings:
        raise HTTPException(409, "마스크가 지정된 Moving 이미지가 없습니다")
    missing = [SESSION.images[m]["name"] for m in SESSION.moving_ids()
               if _union_mask(m) is None]
    SESSION.running = True
    threading.Thread(target=_run_registration, args=(lazy, profile),
                     daemon=True).start()
    return {"started": True, "count": len(movings), "skipped_no_mask": missing}


def _result_image(mid: str, kind: str) -> np.ndarray:
    r = SESSION.results.get(mid)
    if not r:
        raise HTTPException(404, "결과 없음")
    if kind == "registered":
        img = r.get("registered_img")
    elif kind == "false_color":
        img = r.get("false_color")
    elif kind == "match_viz":
        img = r.get("match_viz")
    elif kind == "fixed":
        img = get_full(SESSION.fixed_id())
    else:
        raise HTTPException(400)
    if img is None:
        raise HTTPException(404, r.get("reason") or "이미지 없음")
    return img


@app.get("/api/result/{mid}/{kind}")
def result_image(mid: str, kind: str, max_side: int = 1600):
    img = _result_image(mid, kind)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.get("/api/result/{mid}/download")
def result_download(mid: str):
    img = _result_image(mid, "registered")
    fixed_name = os.path.splitext(
        SESSION.images[SESSION.fixed_id()]["name"])[0]
    mov_name = os.path.splitext(SESSION.images[mid]["name"])[0]
    out = os.path.join(SESSION.dir, f"{fixed_name}_R_{mov_name}.jpg")
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(out, "wb") as f:
        f.write(buf.tobytes())
    return FileResponse(out, filename=os.path.basename(out),
                        media_type="image/jpeg")


@app.get("/api/events")
async def sse():
    q: asyncio.Queue = asyncio.Queue()
    with _sub_lock:
        _subs.add(q)

    async def gen():
        try:
            yield "event: hello\ndata: {}\n\n"
            while True:
                try:
                    yield await asyncio.wait_for(q.get(), timeout=25)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            with _sub_lock:
                _subs.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "frontend", "dist")
if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="static")


def main():
    import argparse
    import webbrowser

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(
            f"http://127.0.0.1:{args.port}/")).start()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
