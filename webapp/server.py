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
import hashlib
import base64
import socket
from collections import OrderedDict
from dataclasses import replace

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
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
    register_test,
    register_test_lazy,
)
from anchor_recovery import recommend_anchors, register_anchors

log = logging.getLogger(__name__)
APP_VERSION = "1.5.1-local.12"

SAM2_MAX_SIDE = 1024


def _torch_device() -> str:
    """현재 엔진이 쓰는 가속 장치 — /api/state의 device 필드."""
    from compute_device import current
    return current()


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

_img_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_full_cache: OrderedDict[str, np.ndarray] = OrderedDict()  # bounded original-resolution cache
_FULL_BUDGET = 256 * 1024 * 1024
_preview_cache: OrderedDict[tuple, bytes] = OrderedDict()
_PREVIEW_BUDGET = 64 * 1024 * 1024
_recommend_lock = threading.Lock()
_recommend_cache: OrderedDict[str, dict] = OrderedDict()


def _cached_preview(key, render, media_type="image/jpeg"):
    # Call under SESSION.lock. Store encoded bytes only, never retain old sessions.
    if key not in _preview_cache:
        _preview_cache[key] = render()
    contents = _preview_cache[key]
    _preview_cache.move_to_end(key)
    while len(_preview_cache) > 128 or sum(map(len, _preview_cache.values())) > _PREVIEW_BUDGET:
        _preview_cache.popitem(last=False)
    return Response(contents, media_type=media_type, headers={"Cache-Control": "no-store"})


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
        img = _full_cache[img_id]
        _full_cache.move_to_end(img_id)
        while len(_full_cache) > 4 or sum(a.nbytes for a in _full_cache.values()) > _FULL_BUDGET:
            _full_cache.popitem(last=False)
        return img


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
    _preview_cache.clear()
    _sam_current = None
    _mask_previews.clear()  # Invalid drafts must not keep discarded sessions/history alive.


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


def _work_size(w, h):
    scale = min(1, SAM2_MAX_SIDE / max(w, h))
    return max(1, int(w * scale)), max(1, int(h * scale))


def _cache_work(img_id, work):
    _img_cache[img_id] = work
    _img_cache.move_to_end(img_id)
    while len(_img_cache) > 8 or sum(a.nbytes for a in _img_cache.values()) > 24 * 1024**2:
        _img_cache.popitem(last=False)


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
            _cache_work(img_id, full)
        _img_cache.move_to_end(img_id)
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
_sam_current: tuple | None = None  # session, image, edit revision, source path
_sam_features = OrderedDict()
_SAM_FEATURE_BUDGET = 96 * 1024**2


def _get_sam():
    global _sam
    if _sam is None:
        from sam2_mask import load_sam2_predictor
        _sam = load_sam2_predictor()
    return _sam


def _feature_size(value):
    if isinstance(value, dict):
        return sum(_feature_size(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_feature_size(v) for v in value)
    if hasattr(value, 'numel'):
        return value.numel() * value.element_size()
    return getattr(value, 'nbytes', 0)


def _remember_sam(key, sam):
    features = getattr(sam, '_features', None)
    if features is None:
        return
    _sam_features[key] = (features, sam._orig_hw, str(sam.device), _feature_size(features))
    _sam_features.move_to_end(key)
    while len(_sam_features) > 4 or sum(v[3] for v in _sam_features.values()) > _SAM_FEATURE_BUDGET:
        _sam_features.popitem(last=False)


def _sam_select(img_id: str, work=None, key=None) -> None:
    global _sam_current
    if key is None:
        im = _require_image(img_id)
        key = (SESSION.dir, img_id, im['revision'], im['path'])
    if _sam_current != key:
        from sam2_mask import sam_set_image
        sam = _get_sam()
        cached = _sam_features.get(key)
        if cached and cached[2] == str(sam.device):
            sam._features, sam._orig_hw = cached[:2]
            sam._is_image_set, sam._is_batch = True, False
            _sam_features.move_to_end(key)
        else:
            sam_set_image(sam, get_work(img_id) if work is None else work, feat_cache=_sam_features)
            _remember_sam(key, sam)
        _sam_current = key


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
    S = _pixel_scale(work.shape[1] / im['full_w'], work.shape[0] / im['full_h'])
    M = S @ np.asarray(im["G"]) @ np.linalg.inv(part["G"])
    return cv2.warpAffine(part["mask"].astype(np.uint8), M[:2], (work.shape[1], work.shape[0]), flags=cv2.INTER_NEAREST).astype(bool)


def _freeze_mask(img_id, mask):
    if mask is None or isinstance(mask, dict):
        return mask
    im = SESSION.images[img_id]
    h, w = im['full_h'], im['full_w']
    S = _pixel_scale(mask.shape[1] / w, mask.shape[0] / h)
    return {"mask": mask.copy(), "G": S @ np.asarray(im["G"]), "revision": im["revision"]}


def _predict_mask(img_id: str) -> None:
    st = _mask_state(img_id)
    if not st["points"]:
        st["current"] = None
        return
    im = _require_image(img_id)
    st['current'] = _infer_mask(img_id, get_work(img_id), st['points'], (SESSION.dir, img_id, im['revision'], im['path']))


def _infer_mask(img_id, work, points, key):
    """Only this compute boundary owns SAM; it never acquires the session lock."""
    from sam2_mask import sam_predict
    pts = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
    lbl = np.array([p['label'] for p in points], dtype=np.int32)
    with _sam_lock:
        _sam_select(img_id, work, key)
        sam = _get_sam()
        masks, scores, _ = sam_predict(sam, work, pts, lbl, feat_cache=_sam_features)
        _remember_sam(key, sam)
    return masks[int(np.argmax(scores))].astype(bool)


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


def _mask_overlay_png(img_id: str, state=None) -> bytes:
    """현재(연두) + 확정(파랑) 마스크를 RGBA PNG로."""
    st = _mask_state(img_id) if state is None else state
    h, w = get_work(img_id).shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for m in st["confirmed"]:
        rgba[_project_mask(img_id, m)] = (60, 120, 255, 110)
    if st["current"] is not None:
        cur = _project_mask(img_id, st["current"])
        rgba[cur] = (180, 230, 45, 130)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return buf.tobytes()


# ── FastAPI ────────────────────────────────────────

app = FastAPI(title="dkp-registrator-web")


@app.get("/api/app")
def app_identity():
    return {"application": "dkp-registrator", "version": APP_VERSION,
            "build": APP_BUILD}


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
            im = SESSION.images[i]
            w, h = _work_size(im['full_w'], im['full_h'])
            return {
                "id": i, "role": im["role"], "name": im["name"],
                "w": w, "h": h,
                "full_w": im["full_w"], "full_h": im["full_h"],
                "source_w": im["source_w"], "source_h": im["source_h"],
                "revision": im["revision"], "G": im["G"], "edits": im["edits"],
                "n_objects": len(st.get("confirmed", [])),
                "has_current": st.get("current") is not None,
                "mask_ready": bool(st.get('confirmed')) or st.get('current') is not None,
                "mask_rev": st.get("rev", 0), "mask_points": st.get("points", []),
                "result": _result_summary(SESSION.display_result(i)),
            }
        return {"version": APP_VERSION, "images": [img_info(i) for i in SESSION.order],
                "capabilities": {"anchor_recovery": True, "unmasked_regional_matching": True,
                                 "adaptive_sliding_windows": True},
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
        "fixed_name": r.get("fixed_name", ""),
        "different_reference": r.get("fixed_id") != SESSION.fixed_id(),
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
                     "fixed_id": r["previous"]["fixed_id"], "fixed_name": r["previous"].get("fixed_name", ""), "fixed_revision": r["previous"]["fixed_revision"]}
                    if r.get("previous") else None,
        "status": r.get("status"), "gate": r.get("gate"),
        "label": r.get("label"), "reason": r.get("reason"),
        "n_inlier": m.get("n_inlier"), "inlier_ratio": m.get("inlier_ratio"),
        "reproj_median": m.get("reproj_median"),
        "rotation_deg": m.get("rotation_deg"), "scale": m.get("scale"),
        "anchor_residuals": m.get("anchor_residuals", []),
        "reference_groups": m.get("reference_groups", []),
        "reference_conflict": bool(m.get("reference_conflict")),
        "validation": m.get("validation"),
        "regional": m.get("regional"),
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


def _import_photo(session, contents, filename):
    from PIL import Image, ImageOps
    name = os.path.basename((filename or 'photo.png').replace('\\', '/'))
    digest = hashlib.sha256(contents).hexdigest()
    def existing():
        return next((i for i, im in session.images.items()
                     if im['name'] == name and im.get('source_digest') == digest), None)
    with session.lock:
        duplicate = existing()
        if duplicate:
            return duplicate, False
    with Image.open(io.BytesIO(contents)) as original:
        fmt = original.format
        if fmt not in ('JPEG', 'PNG'):
            raise ValueError('JPEG/PNG 사진을 선택하세요')
        w, h = original.size
        orientation = original.getexif().get(274,1)
        target = _work_size(w,h)
        if orientation in (5,6,7,8):
            w, h = h, w
        original.draft('RGB',target)  # JPEG decoder can reduce before allocating full pixels.
        original.thumbnail(target,Image.Resampling.BOX)
        if original.size != target:
            original = original.resize(target,Image.Resampling.BOX)
        oriented = ImageOps.exif_transpose(original)
        work = np.array(oriented.convert('RGB'))
    img_id = uuid.uuid4().hex
    path = os.path.join(session.dir, img_id + ('.jpg' if fmt == 'JPEG' else '.png'))
    with session.lock:
        if session is not SESSION:
            raise HTTPException(409, '작업 세션이 바뀌었습니다. 사진을 다시 추가하세요.')
        _require_idle()
        duplicate = existing()  # Concurrent imports can finish decoding together.
        if duplicate:
            return duplicate, False
        with open(path, 'wb') as out:
            out.write(contents)  # Keep source bytes without recompression.
        session.images[img_id] = {
            'role': 'moving', 'name': name, 'source_digest': digest,
            'path': path, 'source_path': path, 'source_w': w, 'source_h': h,
            'full_w': w, 'full_h': h, 'revision': 0, 'edits': {}, 'G': np.eye(3).tolist(),
        }
        _cache_work(img_id, work)
        session.order.append(img_id)
        if session.fixed_id() is None:
            session.set_fixed(img_id)
        session.revision += 1
        session.history.redo.clear()
    return img_id, True


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), role: str | None = None) -> dict:
    added, rejected, skipped = [], [], []
    session = SESSION
    for f in files:
        contents = await f.read()
        try:
            image_id, is_new = await asyncio.to_thread(_import_photo, session, contents, f.filename)
            if is_new:
                added.append(image_id)
            else:
                skipped.append({'name': f.filename, 'id': image_id})
        except HTTPException:
            raise
        except Exception as e:
            rejected.append({'name': f.filename, 'reason': str(e)})
    return {"added": added, "ids": added, "rejected": rejected, "skipped": skipped, "fixed_id": SESSION.fixed_id()}


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
        SESSION.displayed_results = {mid: fid for mid, fid in SESSION.displayed_results.items() if mid != img_id and fid != img_id}
        for results in SESSION.result_pairs.values():
            results.pop(img_id, None)
        _record("사진 삭제", img_id, before)
        _invalidate_images()
        return {"ok": True}


@app.get("/api/image/{img_id}")
def serve_image(img_id: str, max_side: int = SAM2_MAX_SIDE):
    with SESSION.lock:
        im = _require_image(img_id)
        max_side = max(1, min(max_side, SAM2_MAX_SIDE))
        key = (SESSION.dir, "image", img_id, im['path'], max_side)
        return _cached_preview(key, lambda: _preview_bytes(get_work(img_id), max_side, quality=88))


@app.get("/api/mask/{img_id}/overlay")
def mask_overlay(img_id: str):
    with SESSION.lock:
        _require_image(img_id)
        # Session revision is monotonic even when undo restores an older mask rev.
        key = (SESSION.dir, "mask", img_id, SESSION.revision)
        return _cached_preview(key, lambda: _mask_overlay_png(img_id), "image/png")


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
            except Exception as exc:
                SESSION.masks[img_id] = before["masks"].get(img_id, {"points": [], "confirmed": [], "current": None, "rev": 0})
                log.exception("SAM mask prediction failed")
                raise HTTPException(503, f"마스크 생성 실패: {type(exc).__name__}: {exc}. 최초 사용 시 모델 다운로드를 위한 인터넷 연결을 확인하세요.") from exc
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


_mask_previews = OrderedDict()


def _preview_mask(img_id, points):
    with SESSION.lock:
        _require_idle()
        session = SESSION
        im = _require_image(img_id)
        revision = session.revision
        key = (session.dir, img_id, im['revision'], im['path'])
        work = get_work(img_id)
        h, w = work.shape[:2]
        if not points or len(points) > 100:
            raise HTTPException(422, "미리보기 점은 1~100개를 사용할 수 있습니다")
        for p in points:
            try:
                valid = np.isfinite([p['x'], p['y']]).all() and 0 <= p['x'] < w and 0 <= p['y'] < h and p['label'] in (0, 1)
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                raise HTTPException(422, "마스크 점이 사진 범위 밖입니다")
    try:
        mask = _infer_mask(img_id, work, points, key)
    except Exception as exc:
        log.exception('SAM preview failed')
        raise HTTPException(503, f'마스크 미리보기 실패: {type(exc).__name__}: {exc}. 최초 실행 시 모델 다운로드에 인터넷이 필요합니다.') from exc
    with session.lock:
        if session is not SESSION or revision != session.revision:
            raise HTTPException(409, '사진이나 작업 상태가 바뀌었습니다. 다시 클릭하세요.')
        _require_idle()
        st = dict(_mask_state(img_id), current=mask)
        token = uuid.uuid4().hex
        frozen = _freeze_mask(img_id, mask)
        overlay = base64.b64encode(_mask_overlay_png(img_id, st)).decode('ascii')
        _mask_previews[token] = (session, img_id, revision, frozen)
        while len(_mask_previews) > 8:
            _mask_previews.popitem(last=False)
        return {'token': token, 'overlay': 'data:image/png;base64,' + overlay}


@app.post("/api/mask/{img_id}/preview")
async def mask_preview(img_id: str, points: list = Body(embed=True)):
    return await asyncio.to_thread(_preview_mask, img_id, points)


def _commit_mask_preview(img_id, token):
    with SESSION.lock:
        _require_idle()
        draft = _mask_previews.pop(token, None)
        if not draft or draft[0] is not SESSION or draft[1] != img_id or draft[2] != SESSION.revision:
            raise HTTPException(409, "사진이나 작업 상태가 바뀌었습니다. 다시 클릭한 뒤 Z로 확정하세요.")
        before = SESSION.snapshot()
        st = _mask_state(img_id)
        st['confirmed'].append(draft[3])
        st['points'], st['current'] = [], None
        st['rev'] += 1
        _record("마스크 확정", img_id, before)
        return {"points": [], "n_objects": len(st['confirmed'])}


@app.post("/api/mask/{img_id}/action")
async def mask_action(img_id: str, action: str = Body(embed=True), draft_token: str | None = Body(None)):
    if action == "confirm" and draft_token:
        return await asyncio.to_thread(_commit_mask_preview, img_id, draft_token)
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
    im = _require_image(img_id)
    h, w = im['full_h'], im['full_w']
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
def put_anchors(mid: str, pairs: list[dict] = Body(embed=True), base_revision: int = Body(embed=True), fixed_id: str = Body(embed=True), input_token: str | None = Body(default=None, embed=True)):
    with SESSION.lock:
        _require_idle()
        st = _anchor_state(mid)
        if fixed_id != SESSION.fixed_id() or base_revision != st["revision"]:
            raise HTTPException(409, "앵커 상태가 변경됐습니다. 다시 선택하세요")
        if input_token is not None and input_token != _recommend_token(mid):
            raise HTTPException(409, "사진 또는 마스크가 바뀌었습니다. 앵커를 다시 추천받으세요.")
        if len(pairs) > 100:
            raise HTTPException(422, "앵커는 100쌍까지 사용할 수 있습니다")
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
                        "enabled": bool(p.get("requested_enabled", p.get("enabled", True))),
                        "source": "automatic" if p.get("source") == "automatic" else "manual",
                        "group": str(p.get("group", "manual"))[:40]} for p in pairs]
        st["revision"] += 1
        _record("앵커 변경", mid, before)
        return get_anchors(mid)


def _recommend_token(mid):
    fid = SESSION.fixed_id()
    st = _anchor_state(mid)
    values = [SESSION.dir, fid, mid, st['revision']]
    for iid in (fid, mid):
        values += [SESSION.images[iid]['revision'], SESSION.masks.get(iid, {}).get('rev', 0)]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def _recommend_regions(iid):
    st = _mask_state(iid)
    parts = list(st['confirmed'])
    if st['current'] is not None:
        parts.append(st['current'])
    if not parts:
        return []
    # Stored selections retain the frame in which they were drawn. Bring each
    # to the current image frame before combining (including edited photos).
    union = np.zeros(get_work(iid).shape[:2], dtype=bool)
    for part in parts:
        union |= _project_mask(iid, part)
    return [union]


@app.post('/api/anchors/{mid}/recommend')
def suggest_anchors(mid: str, fixed_id: str = Body(embed=True)):
    session = SESSION
    with session.lock:
        _require_idle()
        if fixed_id != session.fixed_id():
            raise HTTPException(409, '기준 사진이 바뀌었습니다. 다시 시도하세요.')
        token = _recommend_token(mid)
        if token in _recommend_cache:
            _recommend_cache.move_to_end(token)
            return _recommend_cache[token]
        regions = [_recommend_regions(i) for i in (fixed_id, mid)]
        if not all(regions):
            raise HTTPException(422, '기준 사진과 현재 사진에서 구조물을 마스크로 선택하고 Z로 확정하세요.')
        images = [get_work(i) for i in (fixed_id, mid)]
        # Each work image may have slightly different x/y sampling due to rounding.
        frames = [np.linalg.inv(_pixel_scale(im.shape[1]/session.images[i]['full_w'],
                     im.shape[0]/session.images[i]['full_h']) @ np.asarray(session.images[i]['G']))
                  for i, im in zip((fixed_id, mid), images)]
        revision = _anchor_state(mid)['revision']
        if not _recommend_lock.acquire(blocking=False):
            raise HTTPException(409, '다른 앵커 추천이 진행 중입니다. 잠시 뒤 다시 시도하세요.')
    try:
        # SAM and recommendation inference do not run on the device concurrently.
        with _sam_lock:
            suggestions, missing = recommend_anchors(*images, *regions)
        pairs = []
        for p in suggestions:
            points = [(frame @ np.array([*p[side], 1.]))[:2].tolist()
                      for frame, side in zip(frames, ('fixed', 'moving'))]
            pairs.append({'id': 'auto-' + uuid.uuid4().hex, 'fixed': points[0], 'moving': points[1],
                          'enabled': True, 'source': 'automatic', 'group': p['group']})
        with session.lock:
            if SESSION is not session or fixed_id != session.fixed_id() or mid not in session.images or token != _recommend_token(mid):
                raise HTTPException(409, '사진 또는 마스크가 바뀌었습니다. 앵커를 다시 추천받으세요.')
            response = {'pairs': pairs, 'missing': missing, 'token': token,
                        'fixed_id': fixed_id, 'revision': revision}
            _recommend_cache[token] = response
            while len(_recommend_cache) > 4:
                _recommend_cache.popitem(last=False)
            return response
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(503, f'앵커를 추천하지 못했습니다. {exc}')
    finally:
        _recommend_lock.release()


def _png_response(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(500, "PNG 인코딩 실패")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/image/{img_id}/source")
def image_source(img_id: str):
    im = _require_image(img_id)
    media = 'image/png' if im['source_path'].lower().endswith('.png') else 'image/jpeg'
    return FileResponse(im['source_path'], media_type=media, headers={'Cache-Control':'private, max-age=3600'})


@app.get("/api/image/{img_id}/source-preview")
def image_source_preview(img_id: str):
    from PIL import Image, ImageOps
    with SESSION.lock:
        im = _require_image(img_id)
        key = (SESSION.dir, 'source-preview', im['source_path'])
        def render():
            with Image.open(im['source_path']) as original:
                original.draft('RGB',(1600,1600))
                preview = ImageOps.exif_transpose(original)
                preview.thumbnail((1600,1600),Image.Resampling.BILINEAR)
                encoded = io.BytesIO()
                preview.convert('RGB').save(encoded,format='JPEG',quality=94)
                return encoded.getvalue()
        response = _cached_preview(key,render,'image/jpeg')
        response.headers['X-Source-Width'] = str(im['source_w'])
        response.headers['X-Source-Height'] = str(im['source_h'])
        return response


def _crop_region(img, x, y, width, height):
    if width < 1 or height < 1 or width > 2048 or height > 2048:
        raise HTTPException(422, "확대 영역 크기는 1~2048 픽셀입니다")
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + width), min(h, y + height)
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(422, "확대 영역이 사진 밖입니다")
    return img[y0:y1, x0:x1]


def _region(img, x, y, width, height):
    return _png_response(_crop_region(img, x, y, width, height))


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
        old_h, old_w = im['full_h'], im['full_w']
        work_h, work_w = get_work(img_id).shape[:2]
        source_from_work = np.linalg.inv(np.asarray(im["G"])) @ _pixel_scale(old_w / work_w, old_h / work_h)
        source_points = [(source_from_work @ np.array([p["x"], p["y"], 1]), p["label"]) for p in st["points"]]
        path = os.path.join(SESSION.dir, f"{img_id}-{uuid.uuid4().hex}.png")
        with open(path,'wb') as out:
            out.write(contents)  # Validated PNG bytes need no second full-size compression.
        im.update(path=path, revision=SESSION.revision + 1, G=G.tolist(), edits=meta["edits"],
                  full_w=rgb.shape[1], full_h=rgb.shape[0])
        _invalidate_images()
        _cache_work(img_id, cv2.resize(rgb, _work_size(rgb.shape[1],rgb.shape[0]), interpolation=cv2.INTER_AREA))
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
                anchor_groups = []
                for pair in get_anchors(mid)["pairs"]:
                    if pair.get("enabled", True):
                        anchors.append(tuple(_project_point(fixed_id, pair["fixed"])) + tuple(_project_point(mid, pair["moving"])))
                        anchor_groups.append(pair.get('group', 'manual') if pair.get('source') == 'automatic' else 'manual:' + pair['id'])
                def cb(cur, total, label):
                    _job_event(job, "lazy", moving_id=mid, lazy_cur=cur, lazy_total=total, lazy_label=label)
                fn = register_test_lazy if lazy else register_test
                pair_cfg = replace(cfg, unmasked_refinement=(cfg.unmasked_refinement
                                   and fmask_real is None and mmask_real is None))
                kw = {"cfg": pair_cfg, "anchor_points": anchors}
                if lazy:
                    kw["progress_callback"] = cb
                if job.get('anchor_only'):
                    entry = register_anchors(fixed_full, m_full, anchors, anchor_groups)
                else:
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
                prev = session.display_result(mid)
                failed = entry.get("status") == "fail"
                kept = bool(failed and prev and prev.get("registered_img") is not None)
                if kept:
                    prev = snapshot(prev)
                    prev.update(latest_attempt_failed=True, latest_attempt_reason=entry.get("reason") or "품질 기준 미달")
                    session.result_pairs[prev['fixed_id']][mid] = prev
                else:
                    entry.update(latest_attempt_failed=failed, latest_attempt_reason=entry.get("reason") if failed else None)
                    if prev and prev.get("registered_img") is not None:
                        entry["previous"] = {k: v for k, v in snapshot(prev).items() if k != "previous"}
                    results[mid] = entry
                    session.displayed_results[mid] = fixed_id
                job["done"] += 1
                job["items"][mid] = "failed" if failed else "done"
                session.record("정합 재시도" if kept else "정합 결과", mid, before)
                _job_event(job, "one_done", id=mid, moving_id=mid, summary=_result_summary(session.display_result(mid)), kept=kept)
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
                 only: list[str] | None = Body(default=None, embed=True),
                 anchor_only: bool = Body(default=False, embed=True),
                 expected_fixed: str | None = Body(default=None, embed=True),
                 expected_anchor_revision: int | None = Body(default=None, embed=True)) -> dict:
    with SESSION.lock:
        _require_idle()
        if _recommend_lock.locked():
            raise HTTPException(409, '앵커 추천이 진행 중입니다. 완료 후 정합하세요.')
        if profile not in PROFILES:
            raise HTTPException(422, "지원하는 프로필은 기본/엄격입니다")
        if not SESSION.fixed_id():
            raise HTTPException(409, "기준 사진을 추가하세요")
        movings = [m for m in SESSION.moving_ids() if only is None or m in set(only)]
        if not movings:
            raise HTTPException(422, "비교할 사진을 선택하세요")
        if anchor_only:
            if len(movings) != 1 or expected_fixed != SESSION.fixed_id():
                raise HTTPException(409, '현재 사진과 기준 사진을 다시 확인하세요.')
            st = get_anchors(movings[0])
            if expected_anchor_revision != st['revision']:
                raise HTTPException(409, '앵커가 바뀌었습니다. 다시 확인하세요.')
            if sum(bool(p.get('enabled')) for p in st['pairs']) < 2:
                raise HTTPException(422, '서로 떨어진 앵커를 2쌍 이상 지정하세요.')
        job_id = uuid.uuid4().hex
        SESSION.job = {"job_id": job_id, "target_ids": movings.copy(), "fixed_id": SESSION.fixed_id(),
                       "anchor_only": anchor_only,
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
    r = SESSION.display_result(mid)
    if previous and r:
        r = r.get("previous")
    if not r:
        raise HTTPException(404, "결과 없음")
    if kind == "registered":
        img = r.get("registered_img")
    elif kind == "false_color":
        img = _display_false_color(r["fixed_img"], r["registered_img"]) if r.get("registered_img") is not None else None
    elif kind == "match_viz":
        img = r.get("match_viz")
    elif kind == "fixed":
        img = r.get("fixed_img")
    else:
        raise HTTPException(400)
    if img is None:
        raise HTTPException(404, r.get("reason") or "이미지 없음")
    return img


def _display_false_color(fixed, registered):
    # Display images are RGB uint8. Do not reinterpret a dark 0/1 crop as 0..1 floats.
    out = fixed.copy()
    gray = cv2.cvtColor(registered, cv2.COLOR_RGB2GRAY)
    out[:, :, 0] = gray
    out[:, :, 2] = gray
    return out


# /{kind} 보다 반드시 먼저 등록 — FastAPI는 등록 순서로 매칭하므로 뒤에 두면
# /download 요청이 kind="download"로 잡혀 400이 난다
# ── GPU 가속 (선택 설치) ───────────────────────────

_gpu_state = {"installing": False, "phase": "", "done": 0, "total": 0, "error": ""}


@app.get("/api/gpu")
def gpu_status() -> dict:
    import gpu_setup
    import compute_device
    models = {}
    for name, module, attr in [('매칭', 'matching', '_loftr_model'), ('마스크', 'sam2_mask', '_sam2_predictor')]:
        model = getattr(sys.modules.get(module), attr, None)
        if model is not None:
            model = getattr(model, 'model', model)
            models[name] = str(next(model.parameters()).device)
    return {"device": _torch_device(), "gpu_name": gpu_setup.gpu_name(),
            "installed": gpu_setup.installed(), "frozen": getattr(sys, "frozen", False),
            "mode": compute_device.mode(), "accelerator": compute_device.accelerator(),
            "platform": sys.platform, "models": models,
            **_gpu_state}


@app.post('/api/gpu/device')
def gpu_device(mode: str = Body(embed=True)):
    global _sam, _sam_current
    import compute_device
    import gc
    import torch
    with SESSION.lock:
        _require_idle()
        if _gpu_state['installing'] or not _recommend_lock.acquire(blocking=False):
            raise HTTPException(409, '가속 설치 또는 앵커 추천이 끝난 뒤 전환하세요.')
        try:
            if not _sam_lock.acquire(blocking=False):
                raise HTTPException(409, '마스크 계산이 끝난 뒤 전환하세요.')
            try:
                try:
                    compute_device.select(mode)
                except ValueError as exc:
                    raise HTTPException(422, str(exc))
                # Free device-bound singletons; reload from cached weights on next use.
                _sam, _sam_current = None, None
                for module, attr in [('matching', '_loftr_model'), ('sam2_mask', '_sam2_predictor')]:
                    loaded = sys.modules.get(module)
                    if loaded is not None:
                        setattr(loaded, attr, None)
                _sam_features.clear()
                _recommend_cache.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            finally:
                _sam_lock.release()
        finally:
            _recommend_lock.release()
    status = gpu_status()
    publish('gpu', status)
    return status


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


_folder_lock = threading.Lock()


@app.post("/api/select_folder")
def select_folder() -> dict:
    if not _folder_lock.acquire(blocking=False):
        raise HTTPException(409, '폴더 선택창이 이미 열려 있습니다. 열린 창을 확인하거나 경로를 직접 입력하세요.')
    try:
        return _choose_folder()
    finally:
        _folder_lock.release()


def _choose_folder() -> dict:
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
                               capture_output=True, timeout=300,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            path = r.stdout.decode("utf-8", "replace").strip()
    except (OSError, subprocess.TimeoutExpired):
        raise HTTPException(500, "폴더 선택 다이얼로그 실패")
    if r.returncode != 0:
        if sys.platform == 'darwin' and b'-128' in r.stderr:
            return {'path': None}
        raise HTTPException(500, '폴더 선택창을 열지 못했습니다. 폴더 경로를 직접 입력하거나 파일 다운로드를 사용하세요.')
    path = path.splitlines()[-1].strip() if path else ""
    return {"path": path if os.path.isdir(path) else None}


def _encode_result_jpg(mid: str) -> tuple[str, bytes]:
    """(파일명, JPEG 바이트) — 개별 다운로드와 동일 규칙."""
    with SESSION.lock:
        img = _result_image(mid, "registered")
        fixed_name = os.path.splitext(SESSION.display_result(mid)["fixed_name"])[0]
        mov_name = os.path.splitext(SESSION.images[mid]["name"])[0]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise HTTPException(500, "JPEG 인코딩 실패")
    return f"{fixed_name}_R_{mov_name}.jpg", buf.tobytes()


@app.post('/api/export')
def export_results(only: list[str] = Body(embed=True),
                   expected_fixed_id: str = Body(embed=True),
                   expected_results: dict[str, str] = Body(embed=True)):
    import zipfile
    from urllib.parse import quote
    with SESSION.lock:
        if expected_fixed_id != SESSION.fixed_id() or any(
            (SESSION.display_result(mid) or {}).get('id') != expected_results.get(mid)
            for mid in only):
            raise HTTPException(409, '저장할 결과가 바뀌었습니다. 다시 선택해 주세요.')
        targets = list(dict.fromkeys(only))
        if not targets or any(mid not in SESSION.moving_ids() or not SESSION.display_result(mid) for mid in targets):
            raise HTTPException(409, '저장할 정합 결과가 없습니다.')
        if len(targets) == 1:
            name, contents = _encode_result_jpg(targets[0])
            media = 'image/jpeg'
        else:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
                used = set()
                for mid in targets:
                    filename, data = _encode_result_jpg(mid)
                    stem = os.path.splitext(filename)[0]
                    suffix = 2
                    while filename in used:
                        filename = f'{stem}_{suffix}.jpg'; suffix += 1
                    used.add(filename)
                    archive.writestr(filename, data)
            name, contents, media = 'DKP-정합결과.zip', buffer.getvalue(), 'application/zip'
    return Response(contents, media_type=media, headers={
        'Content-Disposition': "attachment; filename*=UTF-8''"+quote(name),
        'Cache-Control':'no-store'})


@app.post("/api/save_results")
def save_results(dir: str = Body(embed=True),
                 only: list[str] | None = Body(default=None, embed=True),
                 expected_fixed_id: str | None = Body(default=None, embed=True),
                 expected_results: dict[str, str] | None = Body(default=None, embed=True)) -> dict:
    """정합 결과 일괄 저장 — 전체(only=None) 또는 선택(only=[id,...])."""
    with SESSION.lock:
        if expected_fixed_id is not None and expected_fixed_id != SESSION.fixed_id():
            raise HTTPException(409, "저장 폴더를 선택하는 동안 기준이 바뀌었습니다. 저장할 결과를 다시 선택해 주세요.")
        if expected_results is not None and any((SESSION.display_result(mid) or {}).get("id") != rid for mid, rid in expected_results.items()):
            raise HTTPException(409, "저장 폴더를 선택하는 동안 결과가 바뀌었습니다. 저장할 결과를 다시 선택해 주세요.")
        if not os.path.isdir(dir):
            raise HTTPException(400, f"폴더가 없습니다: {dir}")
        targets = [m for m in SESSION.moving_ids()
                   if (only is None or m in set(only)) and SESSION.display_result(m)]
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
    return _result_region(mid, kind, x, y, width, height)


@app.get("/api/result/{mid}/previous/region")
def previous_region(mid: str, kind: str = "registered", x: int = 0, y: int = 0, width: int = 512, height: int = 512):
    return _result_region(mid, kind, x, y, width, height, previous=True)


def _result_region(mid, kind, x, y, width, height, previous=False):
    if kind not in ("fixed", "registered", "false_color"):
        raise HTTPException(422, "지원하지 않는 확대 이미지")
    with SESSION.lock:
        if kind == "false_color":
            fixed = _crop_region(_result_image(mid, "fixed", previous), x, y, width, height)
            registered = _crop_region(_result_image(mid, "registered", previous), x, y, width, height)
            return _png_response(_display_false_color(fixed, registered))
        return _region(_result_image(mid, kind, previous), x, y, width, height)


@app.get("/api/result/{mid}/previous/{kind}")
def previous_result_image(mid: str, kind: str, max_side: int = 1600):
    return _result_preview(mid, kind, max_side, previous=True)


@app.post('/api/result/{mid}/restore-previous')
def restore_previous_result(mid: str, result_id: str = Body(embed=True)):
    with SESSION.lock:
        _require_idle()
        _require_image(mid)
        current = SESSION.display_result(mid)
        if not current or current.get('id') != result_id or not current.get('previous'):
            raise HTTPException(409, '표시 중인 결과가 바뀌었습니다. 다시 확인하세요.')
        before = SESSION.snapshot()
        previous = snapshot(current['previous'])
        previous['previous'] = {k: v for k, v in snapshot(current).items() if k != 'previous'}
        SESSION.result_pairs.setdefault(previous['fixed_id'], {})[mid] = previous
        SESSION.displayed_results[mid] = previous['fixed_id']
        _record('이전 정합 결과 복원', mid, before)
        return _result_summary(previous)


@app.post("/api/result/{mid}/review")
def review_result(mid: str, result_id: str = Body(embed=True), status: str = Body(embed=True)):
    with SESSION.lock:
        r = SESSION.display_result(mid)
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
    return _result_preview(mid, kind, max_side)


def _result_preview(mid, kind, max_side, previous=False):
    with SESSION.lock:
        r = SESSION.display_result(mid)
        if previous and r:
            r = r.get('previous')
        if not r:
            raise HTTPException(404, "결과 없음")
        max_side = max(1, min(max_side, 4096))
        key = (SESSION.dir, "result", mid, r['id'], kind, max_side)
        return _cached_preview(key, lambda: _preview_bytes(_result_image(mid, kind, previous), max_side))


def _preview_bytes(img, max_side, quality=90):
    h, w = img.shape[:2]
    max_side = max(1, min(max_side, 4096))
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "미리보기 생성 실패")
    return buf.tobytes()


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
        r = SESSION.display_result(mid)
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
        SESSION.result_pairs[r['fixed_id']][mid] = r
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
APP_BUILD = APP_VERSION
if DIST:
    with open(os.path.join(DIST, "index.html"), "rb") as _index:
        APP_BUILD += "-" + hashlib.sha256(_index.read()).hexdigest()[:12]
if DIST:
    app.mount("/", StaticFiles(directory=DIST, html=True), name="static")


def choose_listener(preferred_port):
    """Reuse only this build; reserve a new port without touching another session."""
    import urllib.request
    for port in range(preferred_port, min(preferred_port + 30, 65536)):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/app", timeout=1) as response:
                if json.load(response) == app_identity():
                    return port, None
        except Exception:
            pass
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", port))
            listener.listen(128)
            return port, listener
        except OSError:
            listener.close()
    raise RuntimeError("사용할 포트가 없습니다. 실행 중인 Registrator 창을 확인하세요.")


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

    port, listener = choose_listener(args.port)
    url = f"http://127.0.0.1:{port}/"
    if listener is None:
        if not args.no_browser:
            webbrowser.open(url)
        return
    if not args.persist:
        threading.Thread(target=_auto_shutdown_loop, daemon=True,
                         name="auto-exit").start()
    if not args.no_browser:
        def open_ready():
            for _ in range(90):
                try:
                    with urllib.request.urlopen(url + "api/app", timeout=1) as response:
                        if json.load(response) == app_identity():
                            webbrowser.open(url)
                            return
                except Exception:
                    pass
                time.sleep(0.5)
        threading.Thread(target=open_ready, daemon=True).start()
    import uvicorn
    try:
        uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")).run(sockets=[listener])
    finally:
        listener.close()


if __name__ == "__main__":
    main()
