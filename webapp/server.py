"""DKP Registrator 웹 UI 백엔드 — FastAPI.

기존 엔진(register.py/sam2_mask.py/config.py)을 그대로 사용하는 브라우저 UI.
tkinter GUI(main_gui.py)와 병행 제공. 실행: py -3.13 webapp/server.py
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # macOS: 미지원 MPS 연산 per-op CPU 폴백

import asyncio
import io
import json
import logging
import sys
import threading
import time
import uuid
from collections import OrderedDict

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
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
from register import (  # noqa: E402
    _apply_orientation,
    false_color,
    register_test,
    register_test_lazy,
)

log = logging.getLogger(__name__)

SAM2_MAX_SIDE = 1024


def _torch_device() -> str:
    """현재 엔진이 쓰는 가속 장치 — /api/state의 device 필드."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def data_dir() -> str:
    base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~")) \
        if os.name == "nt" else os.path.expanduser(
            "~/Library/Application Support" if sys.platform == "darwin"
            else "~/.local/share")
    d = os.path.join(base, "DKPRegistratorWeb")
    os.makedirs(d, exist_ok=True)
    return d


# ── 세션 (단일 활성) ────────────────────────────────

from webapp.session_state import Session as WorkspaceSession
from webapp.history import HistoryConflict, snapshot
from transform import is_similarity

class Session(WorkspaceSession):
    def __init__(self, root=None):
        super().__init__(root or os.environ.get("DKP_SESSION_ROOT") or data_dir())

SESSION = Session()

_img_cache: dict[str, np.ndarray] = {}   # id → RGB (SAM2_MAX_SIDE 제한)
_full_cache: OrderedDict[str, np.ndarray] = OrderedDict()  # bounded original-resolution cache


def _load_rgb(path: str) -> np.ndarray:
    """EXIF 방향 반영 RGB 로드 (main_gui.load_image_rgb와 동일 정책)."""
    from PIL import Image, ImageOps
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return np.array(img.convert("RGB"))


def get_full(img_id: str) -> np.ndarray:
    with SESSION.lock:
        _require_image(img_id)
        if img_id not in _full_cache:
            _full_cache[img_id] = _load_rgb(SESSION.images[img_id]["path"])
            while len(_full_cache) > 4:
                _full_cache.popitem(last=False)
        _full_cache.move_to_end(img_id)
        return _full_cache[img_id]


def _require_image(img_id):
    if img_id not in SESSION.images:
        raise HTTPException(404, "사진을 찾을 수 없습니다")
    return SESSION.images[img_id]


def _require_idle():
    if SESSION.running:
        raise HTTPException(409, "현재 사진 정합을 마친 뒤 변경할 수 있습니다")


def _invalidate_images():
    global _sam_current
    _full_cache.clear()
    _img_cache.clear()
    _sam_current = None


def _record(label, image_id, before):
    SESSION.record(label, image_id, before)


def _pixel_scale(sx, sy):
    """Pixel-center mapping used by OpenCV resize, including rounded x/y dimensions."""
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])


def _freshness(r):
    if not r:
        return "stale"
    f, m = SESSION.images.get(r.get("fixed_id")), SESSION.images.get(r.get("moving_id"))
    key = (r.get("fixed_id"), r.get("moving_id"))
    same = f and m and f["revision"] == r.get("fixed_revision") and m["revision"] == r.get("moving_revision")
    same = same and r.get("anchor_revision", 0) == SESSION.anchors.get(key, {}).get("revision", 0)
    same = same and r.get("fixed_mask_revision", 0) == SESSION.masks.get(r.get("fixed_id"), {}).get("rev", 0)
    same = same and r.get("moving_mask_revision", 0) == SESSION.masks.get(r.get("moving_id"), {}).get("rev", 0)
    return "current" if same else "stale"


def get_work(img_id: str) -> np.ndarray:
    """SAM2/화면용 축소본 (최대 1024px)."""
    with SESSION.lock:
        if img_id not in _img_cache:
            full = get_full(img_id)
            h, w = full.shape[:2]
            s = SAM2_MAX_SIDE / max(h, w)
            if s < 1:
                full = cv2.resize(full, (max(1, int(w * s)), max(1, int(h * s))),
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
_last_disconnect = time.monotonic()


def _client_count() -> int:
    with _sub_lock:
        return len(_subs)


def _idle_seconds() -> float:
    with _sub_lock:
        if _subs:
            return 0.0
        return time.monotonic() - _last_disconnect


def _auto_shutdown_loop() -> None:
    """브라우저 탭이 모두 닫히고 30초 지나면 종료 (정합 실행 중엔 대기).

    첫 접속 전 90초 유예. --persist 또는 --no-browser 시 비활성.
    """
    started = time.monotonic()
    ever = False
    while True:
        time.sleep(5)
        if _client_count() > 0:
            ever = True
            continue
        if SESSION.running:
            continue  # 정합 도중엔 절대 안 죽음
        if not ever:
            if time.monotonic() - started < 90:
                continue
            os._exit(0)
        if _idle_seconds() > 30:
            log.info("UI 종료 감지 - 서버 종료")
            os._exit(0)


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
    _require_image(img_id)
    return SESSION.masks.setdefault(
        img_id, {"points": [], "confirmed": [], "current": None, "rev": 0})


def _project_mask(img_id, part):
    """Reproject from its immutable generation frame, never from a previous projection."""
    if isinstance(part, np.ndarray):
        return part
    im = SESSION.images[img_id]
    work = get_work(img_id)
    full = get_full(img_id)
    S = _pixel_scale(work.shape[1] / full.shape[1], work.shape[0] / full.shape[0])
    M = S @ np.asarray(im["G"]) @ np.linalg.inv(part["G"])
    return cv2.warpAffine(part["mask"].astype(np.uint8), M[:2], (work.shape[1], work.shape[0]), flags=cv2.INTER_NEAREST).astype(bool)


def _freeze_mask(img_id, mask):
    if mask is None or isinstance(mask, dict):
        return mask
    im = SESSION.images[img_id]
    h, w = get_full(img_id).shape[:2]
    S = _pixel_scale(mask.shape[1] / w, mask.shape[0] / h)
    return {"mask": mask.copy(), "G": S @ np.asarray(im["G"]), "revision": im["revision"]}


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
        u |= _project_mask(img_id, m)
    return (u * 255).astype(np.uint8)


def _mask_overlay_png(img_id: str) -> bytes:
    """현재(노랑) + 확정(파랑) 마스크를 RGBA PNG로."""
    st = _mask_state(img_id)
    h, w = get_work(img_id).shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for m in st["confirmed"]:
        rgba[_project_mask(img_id, m)] = (60, 120, 255, 110)
    if st["current"] is not None:
        cur = _project_mask(img_id, st["current"])
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
    with SESSION.lock:
        def img_info(i):
            st = SESSION.masks.get(i, {})
            work, im = get_work(i), SESSION.images[i]
            return {
                "id": i, "role": im["role"], "name": im["name"],
                "w": work.shape[1], "h": work.shape[0],
                "full_w": im["full_w"], "full_h": im["full_h"],
                "source_w": im["source_w"], "source_h": im["source_h"],
                "revision": im["revision"], "G": im["G"], "edits": im["edits"],
                "n_objects": len(st.get("confirmed", [])),
                "has_current": st.get("current") is not None,
                "mask_ready": _union_mask(i) is not None,
                "mask_rev": st.get("rev", 0), "mask_points": st.get("points", []),
                "result": _result_summary(SESSION.results.get(i)),
            }
        return {"images": [img_info(i) for i in SESSION.order],
                "fixed": SESSION.fixed_id(), "fixed_id": SESSION.fixed_id(),
                "revision": SESSION.revision, "running": SESSION.running,
                "job": snapshot(SESSION.job), "history": SESSION.history.labels(),
                "profiles": list(PROFILES), "device": _torch_device()}


def _result_summary(r: dict | None) -> dict | None:
    if not r:
        return None
    m = r.get("metrics") or {}
    freshness = _freshness(r)
    return {
        "id": r.get("id"), "fixed_id": r.get("fixed_id"),
        "fixed_revision": r.get("fixed_revision"), "moving_revision": r.get("moving_revision"),
        "full_w": r.get("fixed_img", np.empty((0, 0))).shape[1],
        "full_h": r.get("fixed_img", np.empty((0, 0))).shape[0],
        "freshness": freshness,
        "review_status": r.get("review_status", "unreviewed") if freshness == "current" else "needs_work",
        "latest_attempt_failed": bool(r.get("latest_attempt_failed")),
        "latest_attempt_reason": r.get("latest_attempt_reason"),
        "has_previous": r.get("previous") is not None,
        "previous": {"id": r["previous"]["id"], "full_w": r["previous"]["fixed_img"].shape[1],
                     "full_h": r["previous"]["fixed_img"].shape[0],
                     "fixed_id": r["previous"]["fixed_id"], "fixed_revision": r["previous"]["fixed_revision"]}
                    if r.get("previous") else None,
        "status": r.get("status"), "gate": r.get("gate"),
        "label": r.get("label"), "reason": r.get("reason"),
        "n_inlier": m.get("n_inlier"), "inlier_ratio": m.get("inlier_ratio"),
        "reproj_median": m.get("reproj_median"),
        "rotation_deg": m.get("rotation_deg"), "scale": m.get("scale"),
        "anchor_residuals": m.get("anchor_residuals", []),
        "manual_adjusted": bool(r.get("manual_adjusted")),
        "used_mask": bool(r.get("used_mask")), "job_id": r.get("job_id"),
    }


@app.post("/api/reset")
def reset() -> dict:
    global SESSION
    with SESSION.lock:
        _require_idle()
        SESSION = Session()
        _invalidate_images()
    return {"ok": True}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), role: str | None = None) -> dict:
    added, rejected = [], []
    for f in files:
        contents = await f.read()
        with SESSION.lock:
            _require_idle()
            img_id = uuid.uuid4().hex
            path = os.path.join(SESSION.dir, img_id + ".png")
            try:
                from PIL import Image, ImageOps
                original = Image.open(io.BytesIO(contents))
                if original.format not in ("JPEG", "PNG"):
                    raise ValueError("JPEG/PNG 사진을 선택하세요")
                original = ImageOps.exif_transpose(original)
                rgb = np.array(original.convert("RGB"))
                Image.fromarray(rgb).save(path, format="PNG")
            except Exception as e:
                rejected.append({"name": f.filename, "reason": str(e)})
                continue
            SESSION.images[img_id] = {
                "role": "moving", "name": os.path.basename((f.filename or "photo.png").replace("\\", "/")),
                "path": path, "source_path": path, "source_w": rgb.shape[1], "source_h": rgb.shape[0],
                "full_w": rgb.shape[1], "full_h": rgb.shape[0],
                "revision": 0, "edits": {}, "G": np.eye(3).tolist(),
            }
            SESSION.order.append(img_id)
            if SESSION.fixed_id() is None:
                SESSION.set_fixed(img_id)
            SESSION.revision += 1
            SESSION.history.redo.clear()
            added.append(img_id)
    return {"added": added, "ids": added, "rejected": rejected, "fixed_id": SESSION.fixed_id()}


@app.post("/api/fixed")
def set_fixed(image_id: str = Body(embed=True), base_revision: int | None = Body(default=None, embed=True)):
    with SESSION.lock:
        _require_image(image_id)
        if base_revision is not None and base_revision != SESSION.revision:
            raise HTTPException(409, "작업 상태가 바뀌었습니다. 새로 확인하세요")
        if SESSION.running:
            SESSION.pending_fixed = image_id
            SESSION.job["stop_requested"] = True
            return {"queued": True, "image_id": image_id}
        if image_id != SESSION.fixed_id():
            before = SESSION.snapshot()
            SESSION.set_fixed(image_id)
            _record("기준 사진 변경", image_id, before)
        return {"ok": True, "fixed_id": image_id, "revision": SESSION.revision}


@app.post("/api/history/{direction}")
def history_action(direction: str):
    with SESSION.lock:
        _require_idle()
        if direction not in ("undo", "redo"):
            raise HTTPException(400)
        source, dest = (SESSION.history.undo, SESSION.history.redo) if direction == "undo" else (SESSION.history.redo, SESSION.history.undo)
        if not source:
            raise HTTPException(409, "되돌릴 작업이 없습니다")
        command = source[-1]
        try:
            SESSION.restore(command["after" if direction == "undo" else "before"],
                            command["before" if direction == "undo" else "after"])
        except HistoryConflict as e:
            raise HTTPException(409, str(e))
        source.pop()
        dest.append(command)
        _invalidate_images()
        return {"image_id": command["image_id"], "label": command["label"], "revision": SESSION.revision}


@app.post("/api/image/{img_id}/delete")
def delete_image(img_id: str) -> dict:
    with SESSION.lock:
        _require_idle()
        _require_image(img_id)
        before = SESSION.snapshot()
        SESSION.order.remove(img_id)
        SESSION.images.pop(img_id)
        SESSION.masks.pop(img_id, None)
        if SESSION.fixed_id() == img_id:
            SESSION.set_fixed(SESSION.order[0] if SESSION.order else None)
        SESSION.anchors = {k: v for k, v in SESSION.anchors.items() if img_id not in k}
        SESSION.result_pairs.pop(img_id, None)
        for results in SESSION.result_pairs.values():
            results.pop(img_id, None)
        _record("사진 삭제", img_id, before)
        _invalidate_images()
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


def _mutate_mask(img_id, action, point=None):
    # SAM selection, prediction and history commit are one serialized action.
    with SESSION.lock:
        _require_idle()
        st = _mask_state(img_id)
        before = SESSION.snapshot()
        if action == "click":
            x, y, label = point
            h, w = get_work(img_id).shape[:2]
            if not np.isfinite([x, y]).all() or not (0 <= x < w and 0 <= y < h) or label not in (0, 1):
                raise HTTPException(422, "마스크 점이 사진 범위 밖입니다")
            st["points"].append({"x": x, "y": y, "label": label})
            try:
                _predict_mask(img_id)
            except Exception:
                SESSION.masks[img_id] = before["masks"].get(img_id, {"points": [], "confirmed": [], "current": None, "rev": 0})
                raise
        elif action == "confirm":
            if st["current"] is None:
                return {"points": st["points"], "n_objects": len(st["confirmed"])}
            st["confirmed"].append(_freeze_mask(img_id, st["current"]))
            st["points"], st["current"] = [], None
        elif action == "reset":
            st["points"], st["confirmed"], st["current"] = [], [], None
        else:
            raise HTTPException(400, "지원하지 않는 마스크 작업")
        st["current"] = _freeze_mask(img_id, st["current"])
        st["rev"] += 1
        _record({"click": "마스크 점 추가", "confirm": "마스크 확정", "reset": "마스크 초기화"}[action], img_id, before)
        return {"points": st["points"], "n_objects": len(st["confirmed"]), "ts": time.time_ns()}


@app.post("/api/mask/{img_id}/click")
async def mask_click(img_id: str, x: float = Body(embed=True), y: float = Body(embed=True), label: int = Body(embed=True)):
    return await asyncio.to_thread(_mutate_mask, img_id, "click", (x, y, label))


@app.post("/api/mask/{img_id}/action")
async def mask_action(img_id: str, action: str = Body(embed=True)):
    if action == "undo":
        return history_action("undo")
    return await asyncio.to_thread(_mutate_mask, img_id, action)


def _anchor_state(mid):
    _require_image(mid)
    if not SESSION.fixed_id() or mid == SESSION.fixed_id():
        raise HTTPException(409, "비교할 사진을 선택하세요")
    return SESSION.anchors.setdefault((SESSION.fixed_id(), mid), {"pairs": [], "revision": 0})


def _project_point(img_id, point):
    return (np.asarray(SESSION.images[img_id]["G"]) @ np.array([*point, 1.]))[:2]


def _point_visible(img_id, point):
    p = _project_point(img_id, point)
    h, w = get_full(img_id).shape[:2]
    return bool(0 <= p[0] < w and 0 <= p[1] < h)


@app.get("/api/anchors/{mid}")
def get_anchors(mid: str):
    with SESSION.lock:
        st = _anchor_state(mid)
        pairs = snapshot(st["pairs"])
        for p in pairs:
            p["requested_enabled"] = p.get("enabled", True)
            p["enabled"] = p.get("enabled", True) and _point_visible(SESSION.fixed_id(), p["fixed"]) and _point_visible(mid, p["moving"])
        return {"pairs": pairs, "revision": st["revision"], "fixed_id": SESSION.fixed_id()}


@app.put("/api/anchors/{mid}")
def put_anchors(mid: str, pairs: list[dict] = Body(embed=True), base_revision: int = Body(embed=True), fixed_id: str = Body(embed=True)):
    with SESSION.lock:
        _require_idle()
        st = _anchor_state(mid)
        if fixed_id != SESSION.fixed_id() or base_revision != st["revision"]:
            raise HTTPException(409, "앵커 상태가 변경됐습니다. 다시 선택하세요")
        ids = set()
        for p in pairs:
            if not isinstance(p.get("id"), str) or not p["id"] or p["id"] in ids:
                raise HTTPException(422, "잘못된 앵커 ID")
            ids.add(p["id"])
            for side, iid in (("fixed", fixed_id), ("moving", mid)):
                xy = p.get(side)
                if not isinstance(xy, list) or len(xy) != 2:
                    raise HTTPException(422, "잘못된 앵커 좌표")
                try:
                    xy = np.asarray(xy, dtype=float)
                except (ValueError, TypeError):
                    raise HTTPException(422, "잘못된 앵커 좌표")
                im = SESSION.images[iid]
                if not np.isfinite(xy).all() or not (0 <= xy[0] < im["source_w"] and 0 <= xy[1] < im["source_h"]):
                    raise HTTPException(422, "앵커가 원본 사진 범위 밖입니다")
        before = SESSION.snapshot()
        st["pairs"] = [{"id": p["id"], "fixed": p["fixed"], "moving": p["moving"],
                        "enabled": bool(p.get("requested_enabled", p.get("enabled", True)))} for p in pairs]
        st["revision"] += 1
        _record("앵커 변경", mid, before)
        return get_anchors(mid)


def _png_response(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(500, "PNG 인코딩 실패")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/image/{img_id}/source")
def image_source(img_id: str):
    im = _require_image(img_id)
    return _png_response(_load_rgb(im["source_path"]))


def _region(img, x, y, width, height):
    if width < 1 or height < 1 or width > 2048 or height > 2048:
        raise HTTPException(422, "확대 영역 크기는 1~2048 픽셀입니다")
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + width), min(h, y + height)
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(422, "확대 영역이 사진 밖입니다")
    return _png_response(img[y0:y1, x0:x1])


@app.get("/api/image/{img_id}/region")
def image_region(img_id: str, x: int = 0, y: int = 0, width: int = 512, height: int = 512):
    _require_image(img_id)
    return _region(get_full(img_id), x, y, width, height)


@app.post("/api/image/{img_id}/edit")
async def edit_image(img_id: str, image: UploadFile = File(...), metadata: str = Form(...)):
    contents = await image.read()
    try:
        meta = json.loads(metadata)
        G = np.asarray(meta["G"], dtype=float).reshape(3, 3)
        from PIL import Image
        decoded = Image.open(io.BytesIO(contents))
        if decoded.format != "PNG":
            raise ValueError("편집 이미지는 PNG여야 합니다")
        rgb = np.array(decoded.convert("RGB"))
        if not is_similarity(G, allow_reflection=True):
            raise ValueError("사진 비율을 보존해야 합니다")
        if rgb.shape[:2] != (meta["height"], meta["width"]):
            raise ValueError("편집 크기가 일치하지 않습니다")
        if not isinstance(meta["edits"], dict):
            raise ValueError("잘못된 편집 정보")
    except (ValueError, KeyError, TypeError, OSError) as e:
        raise HTTPException(422, str(e))
    with SESSION.lock:
        _require_idle()
        im = _require_image(img_id)
        if meta.get("base_revision") != im["revision"]:
            raise HTTPException(409, "사진이 변경됐습니다. 편집을 다시 여세요")
        before = SESSION.snapshot()
        st = _mask_state(img_id)
        st["current"] = _freeze_mask(img_id, st["current"])
        st["confirmed"] = [_freeze_mask(img_id, p) for p in st["confirmed"]]
        old_h, old_w = get_full(img_id).shape[:2]
        work_h, work_w = get_work(img_id).shape[:2]
        source_from_work = np.linalg.inv(np.asarray(im["G"])) @ _pixel_scale(old_w / work_w, old_h / work_h)
        source_points = [(source_from_work @ np.array([p["x"], p["y"], 1]), p["label"]) for p in st["points"]]
        path = os.path.join(SESSION.dir, f"{img_id}-{uuid.uuid4().hex}.png")
        Image.fromarray(rgb).save(path, format="PNG")
        im.update(path=path, revision=SESSION.revision + 1, G=G.tolist(), edits=meta["edits"],
                  full_w=rgb.shape[1], full_h=rgb.shape[0])
        _invalidate_images()
        nh, nw = get_work(img_id).shape[:2]
        to_work = _pixel_scale(nw / rgb.shape[1], nh / rgb.shape[0]) @ G
        st["points"] = []
        for point, label in source_points:
            p = to_work @ point
            if 0 <= p[0] < nw and 0 <= p[1] < nh:
                st["points"].append({"x": float(p[0]), "y": float(p[1]), "label": label})
        st["rev"] += 1
        _record("기준 사진 편집", img_id, before)
        return {"ok": True, "image_id": img_id, "revision": im["revision"]}


def _full_mask(img_id: str) -> np.ndarray:
    """마스크 미지정 시 전체영역 정합용 전면 마스크 (엔진 무수정 경로)."""
    h, w = get_work(img_id).shape[:2]
    return np.full((h, w), 255, dtype=np.uint8)


def _job_event(job, state, **fields):
    publish("register", {"job_id": job["job_id"], "target_ids": job["target_ids"],
                         "fixed_id": job["fixed_id"], "state": state,
                         "done": job["done"], "total": job["total"], **fields})


def _run_registration(lazy: bool, profile: str, movings: list[str]) -> None:
    session = SESSION
    job = session.job
    fixed_id = job["fixed_id"]
    cfg = get_profile(profile)
    try:
        fixed_full = get_full(fixed_id)
        fmask_real = _union_mask(fixed_id)
        for mid in movings:
            with session.lock:
                if job["stop_requested"]:
                    break
                job["items"][mid] = "running"
                job["moving_id"] = mid
            _job_event(job, "progress", moving_id=mid, name=session.images[mid]["name"])
            used_mask = False
            try:
                m_full = get_full(mid)
                mmask_real = _union_mask(mid)
                used_mask = fmask_real is not None and mmask_real is not None
                fmask = fmask_real if used_mask else _full_mask(fixed_id)
                mmask = mmask_real if used_mask else _full_mask(mid)
                fmask_full = cv2.resize(fmask, (fixed_full.shape[1], fixed_full.shape[0]), interpolation=cv2.INTER_NEAREST)
                mmask_full = cv2.resize(mmask, (m_full.shape[1], m_full.shape[0]), interpolation=cv2.INTER_NEAREST)
                anchors = []
                for pair in get_anchors(mid)["pairs"]:
                    if pair.get("enabled", True):
                        anchors.append(tuple(_project_point(fixed_id, pair["fixed"])) + tuple(_project_point(mid, pair["moving"])))
                def cb(cur, total, label):
                    _job_event(job, "lazy", moving_id=mid, lazy_cur=cur, lazy_total=total, lazy_label=label)
                fn = register_test_lazy if lazy else register_test
                kw = {"cfg": cfg, "anchor_points": anchors}
                if lazy:
                    kw["progress_callback"] = cb
                entry = fn(fixed_full, m_full, fmask_full, mmask_full, **kw)[0]
                if entry.get("M_full") is not None and not is_similarity(entry["M_full"]):
                    raise ValueError("비율을 보존하지 않는 정합 결과를 거절했습니다")
            except Exception as e:
                log.exception("registration failed for %s", mid)
                entry = {"status": "fail", "gate": "none", "reason": str(e), "metrics": {}}
            with session.lock:
                before = session.snapshot()
                entry.update(id=uuid.uuid4().hex, moving_id=mid, fixed_id=fixed_id,
                             fixed_revision=session.images[fixed_id]["revision"],
                             moving_revision=session.images[mid]["revision"],
                             fixed_mask_revision=session.masks.get(fixed_id, {}).get("rev", 0),
                             moving_mask_revision=session.masks.get(mid, {}).get("rev", 0),
                             anchor_revision=session.anchors.get((fixed_id, mid), {}).get("revision", 0),
                             fixed_img=fixed_full, moving_path=session.images[mid]["path"],
                             fixed_name=session.images[fixed_id]["name"], used_mask=used_mask,
                             review_status="unreviewed", job_id=job["job_id"])
                # Derive overlays on demand and retain the pinned source path, avoiding
                # two additional full-resolution arrays per registered photo.
                entry.pop("false_color", None)
                results = session.result_pairs.setdefault(fixed_id, {})
                prev = results.get(mid)
                failed = entry.get("status") == "fail"
                kept = bool(failed and prev and prev.get("registered_img") is not None)
                if kept:
                    prev = snapshot(prev)
                    prev.update(latest_attempt_failed=True, latest_attempt_reason=entry.get("reason") or "품질 기준 미달")
                    results[mid] = prev
                else:
                    entry.update(latest_attempt_failed=failed, latest_attempt_reason=entry.get("reason") if failed else None)
                    if prev and prev.get("registered_img") is not None:
                        entry["previous"] = {k: v for k, v in snapshot(prev).items() if k != "previous"}
                    results[mid] = entry
                job["done"] += 1
                job["items"][mid] = "failed" if failed else "done"
                session.record("정합 재시도" if kept else "정합 결과", mid, before)
                _job_event(job, "one_done", id=mid, moving_id=mid, summary=_result_summary(results[mid]), kept=kept)
    except Exception as e:
        log.exception("registration job failed")
        job["error"] = str(e)
        _job_event(job, "error", detail=str(e))
    finally:
        with session.lock:
            for mid in movings:
                if job["items"][mid] == "queued":
                    job["items"][mid] = "cancelled"
            job["cancelled"] = any(v == "cancelled" for v in job["items"].values())
            job["state"] = "done"
            session.running = False
            if session.pending_fixed:
                before = session.snapshot()
                session.set_fixed(session.pending_fixed)
                session.record("기준 사진 변경", session.pending_fixed, before)
                session.pending_fixed = None
            _job_event(job, "done", cancelled=job["cancelled"], items=job["items"])


@app.post("/api/register")
def run_register(lazy: bool = Body(default=False, embed=True),
                 profile: str = Body(default="normal", embed=True),
                 only: list[str] | None = Body(default=None, embed=True)) -> dict:
    with SESSION.lock:
        _require_idle()
        if profile not in PROFILES:
            raise HTTPException(422, "지원하는 프로필은 기본/엄격입니다")
        if not SESSION.fixed_id():
            raise HTTPException(409, "기준 사진을 추가하세요")
        movings = [m for m in SESSION.moving_ids() if only is None or m in set(only)]
        if not movings:
            raise HTTPException(422, "비교할 사진을 선택하세요")
        job_id = uuid.uuid4().hex
        SESSION.job = {"job_id": job_id, "target_ids": movings.copy(), "fixed_id": SESSION.fixed_id(),
                       "done": 0, "total": len(movings), "state": "running", "stop_requested": False,
                       "cancelled": False, "items": {m: "queued" for m in movings}}
        SESSION.running = True
        threading.Thread(target=_run_registration, args=(lazy, profile, movings), daemon=True).start()
        return {"started": True, "count": len(movings), "job_id": job_id, "target_ids": movings}


@app.post("/api/register/stop")
def stop_registration():
    with SESSION.lock:
        if SESSION.running:
            SESSION.job["stop_requested"] = True
        return {"ok": True, "job_id": (SESSION.job or {}).get("job_id")}


def _result_image(mid: str, kind: str, previous=False) -> np.ndarray:
    r = SESSION.results.get(mid)
    if previous and r:
        r = r.get("previous")
    if not r:
        raise HTTPException(404, "결과 없음")
    if kind == "registered":
        img = r.get("registered_img")
    elif kind == "false_color":
        img = false_color(r["fixed_img"], r["registered_img"]) if r.get("registered_img") is not None else None
    elif kind == "match_viz":
        img = r.get("match_viz")
    elif kind == "fixed":
        img = r.get("fixed_img")
    else:
        raise HTTPException(400)
    if img is None:
        raise HTTPException(404, r.get("reason") or "이미지 없음")
    return img


# /{kind} 보다 반드시 먼저 등록 — FastAPI는 등록 순서로 매칭하므로 뒤에 두면
# /download 요청이 kind="download"로 잡혀 400이 난다
# ── GPU 가속 (선택 설치) ───────────────────────────

_gpu_state = {"installing": False, "phase": "", "done": 0, "total": 0, "error": ""}


@app.get("/api/gpu")
def gpu_status() -> dict:
    import gpu_setup
    return {"device": _torch_device(), "gpu_name": gpu_setup.gpu_name(),
            "installed": gpu_setup.installed(), "frozen": getattr(sys, "frozen", False),
            **_gpu_state}


def _gpu_install_worker() -> None:
    import gpu_setup
    try:
        def on_status(d: dict) -> None:
            _gpu_state.update(phase=d.get("phase", ""), done=d.get("done", 0),
                              total=d.get("total", 0))
            publish("gpu", dict(_gpu_state))
        gpu_setup.install_cuda(on_status)
        _gpu_state.update(phase="done", error="")
    except BaseException as e:
        log.exception("gpu install failed")
        _gpu_state.update(phase="error", error=str(e))
    finally:
        _gpu_state["installing"] = False
        publish("gpu", dict(_gpu_state))


@app.post("/api/gpu/install")
def gpu_install() -> dict:
    """CUDA torch 선택 설치 시작 (백그라운드). 완료 후 앱 재시작 시 적용."""
    import gpu_setup
    if _gpu_state["installing"]:
        raise HTTPException(409, "이미 설치 중입니다")
    if not gpu_setup.gpu_name():
        raise HTTPException(409, "NVIDIA GPU를 찾지 못했습니다")
    _gpu_state.update(installing=True, phase="시작", done=0, total=0, error="")
    threading.Thread(target=_gpu_install_worker, daemon=True).start()
    return {"started": True}


@app.post("/api/gpu/remove")
def gpu_remove() -> dict:
    import gpu_setup
    if _gpu_state["installing"]:
        raise HTTPException(409, "설치 중에는 제거할 수 없습니다")
    gpu_setup.remove_cuda()
    return {"removed": True}


@app.post("/api/select_folder")
def select_folder() -> dict:
    """저장 폴더 선택 — 로컬 네이티브 대화상자 (Windows: IFileOpenDialog, macOS: osascript)."""
    import subprocess
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["osascript", "-e",
                                'POSIX path of (choose folder with prompt "저장 폴더 선택")'],
                               capture_output=True, timeout=300)
            path = r.stdout.decode("utf-8", "replace").strip().rstrip("/")
        else:
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "folder_dialog.ps1")
            r = subprocess.run(["powershell", "-STA", "-NoProfile", "-ExecutionPolicy",
                                "Bypass", "-File", script],
                               capture_output=True, timeout=300)
            path = r.stdout.decode("utf-8", "replace").strip()
    except (OSError, subprocess.TimeoutExpired):
        raise HTTPException(500, "폴더 선택 다이얼로그 실패")
    path = path.splitlines()[-1].strip() if path else ""
    return {"path": path if os.path.isdir(path) else None}


def _encode_result_jpg(mid: str) -> tuple[str, bytes]:
    """(파일명, JPEG 바이트) — 개별 다운로드와 동일 규칙."""
    with SESSION.lock:
        img = _result_image(mid, "registered")
        fixed_name = os.path.splitext(SESSION.results[mid]["fixed_name"])[0]
        mov_name = os.path.splitext(SESSION.images[mid]["name"])[0]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise HTTPException(500, "JPEG 인코딩 실패")
    return f"{fixed_name}_R_{mov_name}.jpg", buf.tobytes()


@app.post("/api/save_results")
def save_results(dir: str = Body(embed=True),
                 only: list[str] | None = Body(default=None, embed=True),
                 expected_fixed_id: str | None = Body(default=None, embed=True),
                 expected_results: dict[str, str] | None = Body(default=None, embed=True)) -> dict:
    """정합 결과 일괄 저장 — 전체(only=None) 또는 선택(only=[id,...])."""
    with SESSION.lock:
        if expected_fixed_id is not None and expected_fixed_id != SESSION.fixed_id():
            raise HTTPException(409, "저장 폴더를 선택하는 동안 기준이 바뀌었습니다. 저장할 결과를 다시 선택해 주세요.")
        if expected_results is not None and any(SESSION.results.get(mid, {}).get("id") != rid for mid, rid in expected_results.items()):
            raise HTTPException(409, "저장 폴더를 선택하는 동안 결과가 바뀌었습니다. 저장할 결과를 다시 선택해 주세요.")
        if not os.path.isdir(dir):
            raise HTTPException(400, f"폴더가 없습니다: {dir}")
        targets = [m for m in SESSION.moving_ids()
                   if (only is None or m in set(only)) and SESSION.results.get(m)]
        if not targets:
            raise HTTPException(409, "저장할 정합 결과가 없습니다")
        saved, failed = [], []
        for mid in targets:
            try:
                name, data = _encode_result_jpg(mid)
                with open(os.path.join(dir, name), "wb") as f:
                    f.write(data)
                saved.append(name)
            except Exception:
                log.exception("save failed: %s", mid)
                failed.append(SESSION.images[mid]["name"])
        return {"saved": len(saved), "failed": failed, "dir": dir}


@app.get("/api/result/{mid}/download")
def result_download(mid: str):
    name, data = _encode_result_jpg(mid)
    out = os.path.join(SESSION.dir, uuid.uuid4().hex + ".jpg")
    with open(out, "wb") as f:
        f.write(data)
    return FileResponse(out, filename=name, media_type="image/jpeg")


@app.get("/api/result/{mid}/region")
def result_region(mid: str, kind: str = "registered", x: int = 0, y: int = 0, width: int = 512, height: int = 512):
    if kind not in ("fixed", "registered"):
        raise HTTPException(422, "지원하지 않는 확대 이미지")
    return _region(_result_image(mid, kind), x, y, width, height)


@app.get("/api/result/{mid}/previous/region")
def previous_region(mid: str, kind: str = "registered", x: int = 0, y: int = 0, width: int = 512, height: int = 512):
    if kind not in ("fixed", "registered"):
        raise HTTPException(422, "지원하지 않는 확대 이미지")
    return _region(_result_image(mid, kind, previous=True), x, y, width, height)


@app.get("/api/result/{mid}/previous/{kind}")
def previous_result_image(mid: str, kind: str, max_side: int = 1600):
    return _preview_response(_result_image(mid, kind, previous=True), max_side)


@app.post("/api/result/{mid}/review")
def review_result(mid: str, result_id: str = Body(embed=True), status: str = Body(embed=True)):
    with SESSION.lock:
        r = SESSION.results.get(mid)
        if not r or r.get("id") != result_id:
            raise HTTPException(409, "결과가 변경됐습니다. 현재 결과를 다시 확인하세요")
        if status not in ("unreviewed", "confirmed", "needs_work"):
            raise HTTPException(422, "지원하지 않는 검토 상태")
        if status == "confirmed" and (_freshness(r) != "current" or r.get("registered_img") is None):
            raise HTTPException(409, "현재 입력으로 정합한 뒤 확인하세요")
        before = SESSION.snapshot()
        r["review_status"] = status
        _record("결과 검토 상태 변경", mid, before)
        return {"ok": True, "result": _result_summary(r)}


@app.get("/api/result/{mid}/{kind}")
def result_image(mid: str, kind: str, max_side: int = 1600):
    return _preview_response(_result_image(mid, kind), max_side)


def _preview_response(img, max_side):
    h, w = img.shape[:2]
    max_side = max(1, min(max_side, 4096))
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.post("/api/result/{mid}/adjust")
def adjust_result(mid: str,
                  dx: float = Body(default=0.0, embed=True),
                  dy: float = Body(default=0.0, embed=True),
                  scale: float = Body(default=1.0, embed=True),
                  rot_deg: float = Body(default=0.0, embed=True),
                  ref_w: float = Body(default=0.0, embed=True),
                  result_id: str | None = Body(default=None, embed=True),
                  reset: bool = Body(default=False, embed=True)) -> dict:
    with SESSION.lock:
        _require_idle()
        r = SESSION.results.get(mid)
        if not r or r.get("M_full") is None:
            raise HTTPException(404, "결과 없음")
        if result_id is not None and result_id != r.get("id"):
            raise HTTPException(409, "정합 결과가 변경됐습니다. 현재 결과에서 다시 조정하세요")
        if not np.isfinite([dx, dy, scale, rot_deg, ref_w]).all() or scale <= 0 or ref_w < 0:
            raise HTTPException(422, "유효한 등방 배율과 이동값을 입력하세요")
        before = SESSION.snapshot()
        r = snapshot(r)
        r["previous"] = {k: v for k, v in snapshot(r).items() if k != "previous"}
        if "M_orig" not in r:
            r["M_orig"] = np.array(r["M_full"], dtype=np.float64).copy()
        fixed = r["fixed_img"]
        h, w = fixed.shape[:2]
        if reset:
            new_M = r["M_orig"].copy()
        else:
            sc = w / ref_w if ref_w else 1.0
            D = np.eye(3, dtype=np.float64)
            D[:2] = cv2.getRotationMatrix2D((w / 2, h / 2), -rot_deg, scale)
            D[0, 2] += dx * sc
            D[1, 2] += dy * sc
            new_M = D @ np.array(r["M_full"], dtype=np.float64)
        if not is_similarity(new_M):
            raise HTTPException(422, "사진 비율을 보존해야 합니다")
        flip, k = r.get("lazy_orientation", (False, 0))
        m_src = _apply_orientation(_load_rgb(r["moving_path"]), flip, k)
        reg = cv2.warpAffine(m_src, new_M[:2, :], (w, h))
        r.update(M_full=new_M, registered_img=reg,
                 manual_adjusted=not np.allclose(new_M, r["M_orig"]),
                 id=uuid.uuid4().hex, review_status="unreviewed")
        # Automatic match metrics no longer describe a manual transform.
        r["metrics"] = {"scale": float(np.sqrt(np.linalg.det(new_M[:2, :2]))),
                        "rotation_deg": float(np.degrees(np.arctan2(new_M[1, 0], new_M[0, 0])))}
        SESSION.results[mid] = r
        _record("정합 미세조정", mid, before)
        return {"ok": True, "manual_adjusted": r["manual_adjusted"], "result_id": r["id"], "ts": time.time_ns()}


@app.get("/api/events")
async def sse():
    q: asyncio.Queue = asyncio.Queue()
    with _sub_lock:
        _subs.add(q)

    async def gen():
        global _last_disconnect
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
                if not _subs:
                    _last_disconnect = time.monotonic()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# 소스 실행: webapp/frontend/dist · frozen(PyInstaller): _MEIPASS/webapp/frontend/dist
_DIST_CANDIDATES = [
    os.path.join(getattr(sys, "_MEIPASS", ""), "webapp", "frontend", "dist"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist"),
]
DIST = next((d for d in _DIST_CANDIDATES if d and os.path.isdir(d)), None)
if DIST:
    app.mount("/", StaticFiles(directory=DIST, html=True), name="static")


def main():
    import argparse
    import urllib.request
    import webbrowser

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--persist", action="store_true",
                    help="브라우저를 닫아도 서버 유지 (개발용)")
    args = ap.parse_args()

    # 이미 실행 중이면 새로 띄우지 않고 브라우저만 (포트 충돌로 옛 세션이 보이는 문제 방지)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/state", timeout=2):
            print("이미 실행 중 - 브라우저만 엽니다")
            if not args.no_browser:
                webbrowser.open(f"http://127.0.0.1:{args.port}/")
            return
    except Exception:
        pass
    if not args.persist:
        threading.Thread(target=_auto_shutdown_loop, daemon=True,
                         name="auto-exit").start()
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(
            f"http://127.0.0.1:{args.port}/")).start()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
