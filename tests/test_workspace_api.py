"""Workspace contracts, using synthetic pixels and a deterministic inference boundary."""
import io
import json
import threading
import time
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from webapp import server as s


def png(a):
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return b.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "SESSION", s.Session(str(tmp_path)))
    s._img_cache.clear()
    s._full_cache.clear()
    monkeypatch.setattr(s, "_torch_device", lambda: "cpu")
    def predict(i):
        st = s._mask_state(i)
        st["current"] = np.ones(s.get_work(i).shape[:2], dtype=bool)
    monkeypatch.setattr(s, "_predict_mask", predict)
    return TestClient(s.app, base_url="http://localhost")


def add(c, n=3):
    a = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
    r = c.post("/api/upload", files=[("files", (f"{i}.png", png(a), "image/png")) for i in range(n)])
    assert r.status_code == 200, r.text
    return r.json()["ids"]


def test_import_session_preserves_existing_directory(tmp_path):
    old = tmp_path / "session"
    old.mkdir()
    (old / "keep.txt").write_text("keep")
    s.Session(str(tmp_path))
    assert (old / "keep.txt").read_text() == "keep"


def test_mask_failure_is_actionable_and_keeps_state(client, monkeypatch):
    image_id = add(client, 1)[0]
    def failure(_):
        raise RuntimeError('model configuration missing')
    monkeypatch.setattr(s, '_predict_mask', failure)
    before = client.get('/api/state').json()
    r = client.post(f'/api/mask/{image_id}/click', json={'x': 20, 'y': 20, 'label': 1})
    assert r.status_code == 503
    assert 'model configuration missing' in r.json()['detail']
    after = client.get('/api/state').json()
    assert after['images'][0]['mask_points'] == []
    assert after['history'] == before['history']


def test_upload_rejects_individually_and_default_fixed(client):
    r = client.post("/api/upload", files=[("files", ("bad.png", b"invalid", "image/png")), ("files", ("ok.png", png(np.zeros((20, 30, 3), np.uint8)), "image/png"))])
    assert r.status_code == 200
    assert len(r.json()["rejected"]) == 1
    first = r.json()["ids"][0]
    add(client, 2)
    state = client.get("/api/state").json()
    assert state["fixed_id"] == first
    assert [i["role"] for i in state["images"]] == ["fixed", "moving", "moving"]


def test_reference_pair_anchors_history_and_conflicts(client):
    a, b, c = add(client)
    pair = {"id": "p1", "fixed": [20, 30], "moving": [21, 31], "enabled": True}
    r = client.put(f"/api/anchors/{b}", json={"fixed_id": a, "base_revision": 0, "pairs": [pair]})
    assert r.status_code == 200, r.text
    assert client.put(f"/api/anchors/{b}", json={"fixed_id": a, "base_revision": 0, "pairs": []}).status_code == 409
    assert client.post("/api/fixed", json={"image_id": c}).status_code == 200
    assert client.get(f"/api/anchors/{b}").json()["pairs"] == []
    assert client.post("/api/history/undo").status_code == 200
    assert client.get(f"/api/anchors/{b}").json()["pairs"][0]["id"] == "p1"
    assert client.get("/api/state").json()["images"][0]["id"] == a


def test_mask_confirm_reset_undo_redo_are_lossless(client):
    a, b, _ = add(client)
    for i in (a, b):
        assert client.post(f"/api/mask/{i}/click", json={"x": 3, "y": 4, "label": 1}).status_code == 200
    client.post(f"/api/mask/{a}/action", json={"action": "confirm"})
    client.post(f"/api/mask/{a}/action", json={"action": "reset"})
    assert s._union_mask(a) is None
    assert s._union_mask(b) is not None
    client.post("/api/history/undo")
    assert len(s._mask_state(a)["confirmed"]) == 1
    client.post("/api/history/undo")
    assert s._mask_state(a)["points"] == [{"x": 3, "y": 4, "label": 1}]
    assert s._mask_state(a)["current"] is not None
    client.post("/api/history/redo")
    assert len(s._mask_state(a)["confirmed"]) == 1


def test_edit_geometry_source_roi_and_stale_revision(client):
    a, b, _ = add(client)
    source = s.get_full(a).copy()
    client.post(f"/api/mask/{a}/click", json={"x": 3, "y": 4, "label": 1})
    G = [[1, 0, -10], [0, 1, -5], [0, 0, 1]]
    meta = {"edits": {"crop": {"x": 10, "y": 5, "w": 40, "h": 30}}, "G": G, "width": 40, "height": 30, "base_revision": 0}
    r = client.post(f"/api/image/{a}/edit", files={"image": ("edit.png", png(source[5:35, 10:50]), "image/png")}, data={"metadata": json.dumps(meta)})
    assert r.status_code == 200, r.text
    assert np.array_equal(np.array(Image.open(io.BytesIO(client.get(f"/api/image/{a}/source").content))), source)
    roi = client.get(f"/api/image/{a}/region?x=2&y=3&width=4&height=5")
    assert np.array_equal(np.array(Image.open(io.BytesIO(roi.content))), source[8:13, 12:16])
    assert s._union_mask(a).shape == (30, 40)
    client.post("/api/history/undo")
    assert np.array_equal(s.get_full(a), source)
    assert s._union_mask(a).shape == (80, 100)
    meta["G"][0][0] = 2
    assert client.post(f"/api/image/{a}/edit", files={"image": ("edit.png", png(source[5:35, 10:50]), "image/png")}, data={"metadata": json.dumps(meta)}).status_code in (409, 422)


def test_relaxed_rejected(client):
    add(client)
    assert client.post("/api/register", json={"profile": "relaxed"}).status_code == 422


def wait_done():
    limit = time.monotonic() + 5
    while s.SESSION.running and time.monotonic() < limit:
        time.sleep(.01)
    assert not s.SESSION.running


def engine_result(f, m, *args, **kw):
    return [{"status": "pass", "gate": "similarity", "registered_img": m.copy(),
             "false_color": f.copy(), "metrics": {"n_inlier": 60}, "M_full": np.eye(3)}]

def test_save_requires_same_reference_and_result_versions(client, monkeypatch, tmp_path):
    a, b, c = add(client)
    monkeypatch.setattr(s, "register_test", engine_result)
    client.post('/api/register', json={'only':[b]})
    wait_done()
    result_id = s.SESSION.results[b]['id']
    body = {'dir':str(tmp_path),'only':[b],'expected_fixed_id':a,'expected_results':{b:result_id}}
    assert client.post('/api/save_results',json=body).status_code == 200
    client.post('/api/register',json={'only':[b]})
    wait_done()
    assert client.post('/api/save_results',json=body).status_code == 409
    client.post('/api/fixed',json={'image_id':c})
    assert client.post('/api/save_results',json=body).status_code == 409


def test_job_isolates_failure_snapshots_targets_and_keeps_previous(client, monkeypatch):
    a, b, c = add(client)
    events = []
    monkeypatch.setattr(s, "publish", lambda event, payload: events.append(payload.copy()))
    calls = []
    def engine(f, m, *args, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise ValueError("synthetic failure")
        return engine_result(f, m)
    monkeypatch.setattr(s, "register_test", engine)
    run = client.post("/api/register", json={"only": [b, c]}).json()
    wait_done()
    assert len(calls) == 2
    assert s.SESSION.results[b]["status"] == "fail"
    old = s.SESSION.results[c]["id"]
    assert all(e["job_id"] == run["job_id"] and e["target_ids"] == [b, c] for e in events)
    assert events[-1]["state"] == "done"
    monkeypatch.setattr(s, "register_test", lambda *a, **k: [{"status": "fail", "reason": "bad attempt"}])
    client.post("/api/register", json={"only": [c]})
    wait_done()
    assert s.SESSION.results[c]["id"] == old
    assert s.SESSION.results[c]["latest_attempt_failed"] is True
    assert client.get("/api/state").json()["images"][2]["result"]["latest_attempt_reason"] == "bad attempt"


def test_reference_change_stops_at_item_boundary(client, monkeypatch):
    a, b, c = add(client)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def engine(f, m, *args, **kw):
        calls.append(kw)
        entered.set()
        assert release.wait(5)
        return engine_result(f, m)
    monkeypatch.setattr(s, "register_test", engine)
    client.post("/api/register", json={})
    assert entered.wait(2)
    assert client.post("/api/fixed", json={"image_id": c}).json()["queued"]
    assert s.SESSION.fixed_id() == a
    assert client.post(f"/api/image/{b}/delete").status_code == 409
    release.set()
    wait_done()
    assert len(calls) == 1
    assert s.SESSION.fixed_id() == c
    assert not s.SESSION.results
    assert s.SESSION.job["items"][c] == "cancelled"
    client.post("/api/history/undo")
    assert s.SESSION.fixed_id() == a
    assert s.SESSION.results[b]["fixed_id"] == a


def test_maskless_anchor_delivery_review_manual_undo_and_pinned_pixels(client, monkeypatch):
    a, b, _ = add(client)
    sent = []
    def engine(f, m, *args, **kw):
        sent.extend(kw["anchor_points"])
        return engine_result(f, m)
    monkeypatch.setattr(s, "register_test", engine)
    client.put(f"/api/anchors/{b}", json={"fixed_id": a, "base_revision": 0, "pairs": [{"id": "one", "fixed": [10, 20], "moving": [11, 21]}]})
    client.post("/api/register", json={"only": [b]})
    wait_done()
    assert sent == [(10, 20, 11, 21)]
    r = s.SESSION.results[b]
    rid = r["id"]
    assert r["used_mask"] is False
    assert client.post(f"/api/result/{b}/review", json={"result_id": "wrong", "status": "confirmed"}).status_code == 409
    assert client.post(f"/api/result/{b}/review", json={"result_id": rid, "status": "confirmed"}).status_code == 200
    before = r["registered_img"].copy()
    assert client.post(f"/api/result/{b}/adjust", json={"result_id": "outdated", "dx": 20}).status_code == 409
    assert client.post(f"/api/result/{b}/adjust", json={"scale": -1}).status_code == 422
    client.post(f"/api/result/{b}/adjust", json={"dx": 2, "rot_deg": 1, "scale": 1.01})
    assert s.SESSION.results[b]["review_status"] == "unreviewed"
    assert s.SESSION.results[b]["id"] != rid
    client.post("/api/history/undo")
    assert np.array_equal(s.SESSION.results[b]["registered_img"], before)
    assert s.SESSION.results[b]["review_status"] == "confirmed"
    original = s.get_full(a).copy()
    meta = {"edits": {"brightness": 20}, "G": np.eye(3).tolist(), "width": 100, "height": 80, "base_revision": 0}
    client.post(f"/api/image/{a}/edit", files={"image": ("edit.png", png(np.zeros_like(original)), "image/png")}, data={"metadata": json.dumps(meta)})
    assert s._result_summary(s.SESSION.results[b])["freshness"] == "stale"
    response = client.get(f"/api/result/{b}/region?kind=fixed&x=0&y=0&width=100&height=80")
    assert np.array_equal(np.array(Image.open(io.BytesIO(response.content))), original)
    assert client.post(f"/api/result/{b}/review", json={"result_id": rid, "status": "confirmed"}).status_code == 409


def test_roi_preserves_original_resolution_and_bounds(client):
    yy, xx = np.indices((1200, 1801))
    a = np.stack([xx % 251, yy % 253, (xx + yy) % 255], axis=-1).astype(np.uint8)
    iid = client.post("/api/upload", files=[("files", ("grid.png", png(a), "image/png"))]).json()["ids"][0]
    assert s.get_work(iid).shape[1] == 1024
    content = client.get(f"/api/image/{iid}/region?x=1700&y=1100&width=200&height=200").content
    assert np.array_equal(np.array(Image.open(io.BytesIO(content))), a[1100:, 1700:])
    assert client.get(f"/api/image/{iid}/region?width=9999").status_code == 422
    assert client.get(f"/api/image/{iid}/region?x=9999").status_code == 422


def test_history_does_not_remove_later_uploaded_photo(client):
    a, b, _ = add(client)
    client.post(f"/api/mask/{a}/click", json={"x": 3, "y": 4, "label": 1})
    extra = add(client, 1)[0]
    client.post("/api/history/undo")
    assert extra in s.SESSION.images
    assert s._union_mask(a) is None


def test_previous_result_is_pinned_and_only_one_generation(client, monkeypatch):
    a, b, _ = add(client)
    monkeypatch.setattr(s, "register_test", engine_result)
    client.post("/api/register", json={"only": [b]})
    wait_done()
    first = s.SESSION.results[b]["id"]
    pixels = s.SESSION.results[b]["registered_img"].copy()
    client.post(f"/api/result/{b}/adjust", json={"dx": 2})
    summary = s._result_summary(s.SESSION.results[b])
    assert summary["has_previous"]
    assert summary["previous"]["id"] == first
    previous = client.get(f"/api/result/{b}/previous/region?kind=registered&width=100&height=80")
    assert np.array_equal(np.array(Image.open(io.BytesIO(previous.content))), pixels)
    client.post(f"/api/result/{b}/adjust", json={"dy": 2})
    assert "previous" not in s.SESSION.results[b]["previous"]


def test_cropped_anchor_reactivates_after_other_pair_edit_and_uncrop(client):
    a, b, _ = add(client)
    pairs = [{"id": "p1", "fixed": [5, 5], "moving": [5, 5]}, {"id": "p2", "fixed": [20, 20], "moving": [20, 20]}]
    client.put(f"/api/anchors/{b}", json={"fixed_id": a, "base_revision": 0, "pairs": pairs})
    original = s.get_full(a).copy()
    def edit(rgb, G):
        return client.post(f"/api/image/{a}/edit", files={"image": ("e.png", png(rgb), "image/png")}, data={"metadata": json.dumps({"G": G, "width": rgb.shape[1], "height": rgb.shape[0], "edits": {}, "base_revision": s.SESSION.images[a]["revision"]})})
    assert edit(original[10:, 10:], [[1, 0, -10], [0, 1, -10], [0, 0, 1]]).status_code == 200
    st = client.get(f"/api/anchors/{b}").json()
    assert not st["pairs"][0]["enabled"]
    st["pairs"][1]["moving"] = [21, 21]
    client.put(f"/api/anchors/{b}", json={"fixed_id": a, "base_revision": st["revision"], "pairs": st["pairs"]})
    assert edit(original, np.eye(3).tolist()).status_code == 200
    assert client.get(f"/api/anchors/{b}").json()["pairs"][0]["enabled"]


def test_new_result_undo_restores_previous_without_transferring_review(client, monkeypatch):
    _, b, _ = add(client)
    monkeypatch.setattr(s, "register_test", engine_result)
    client.post("/api/register", json={"only": [b]})
    wait_done()
    r1 = s.SESSION.results[b]["id"]
    client.post(f"/api/result/{b}/review", json={"result_id": r1, "status": "confirmed"})
    client.post("/api/register", json={"only": [b]})
    wait_done()
    r2 = s.SESSION.results[b]["id"]
    assert r2 != r1 and s.SESSION.results[b]["review_status"] == "unreviewed"
    assert client.post("/api/history/undo").status_code == 200
    assert s.SESSION.results[b]["id"] == r1
    assert s.SESSION.results[b]["review_status"] == "confirmed"
    client.post("/api/history/undo")
    assert s.SESSION.results[b]["id"] == r1
    assert s.SESSION.results[b]["review_status"] == "unreviewed"
    client.post("/api/history/redo")
    assert s.SESSION.results[b]["id"] == r1
    assert s.SESSION.results[b]["review_status"] == "confirmed"
    client.post("/api/history/redo")
    assert s.SESSION.results[b]["id"] == r2
    assert s.SESSION.results[b]["review_status"] == "unreviewed"


def test_manual_result_and_new_auto_result_undo_are_separate(client, monkeypatch):
    _, b, _ = add(client)
    monkeypatch.setattr(s, "register_test", engine_result)
    client.post("/api/register", json={"only": [b]})
    wait_done()
    original = s.SESSION.results[b]
    client.post(f"/api/result/{b}/adjust", json={"dx": 4})
    manual = s.SESSION.results[b]
    client.post("/api/register", json={"only": [b]})
    wait_done()
    client.post("/api/history/undo")
    assert s.SESSION.results[b]["id"] == manual["id"]
    assert np.array_equal(s.SESSION.results[b]["M_full"], manual["M_full"])
    client.post("/api/history/undo")
    assert s.SESSION.results[b]["id"] == original["id"]
    assert np.array_equal(s.SESSION.results[b]["M_full"], original["M_full"])


def test_history_rejects_unexpected_result_identity_without_partial_changes(client, monkeypatch):
    _, b, _ = add(client)
    monkeypatch.setattr(s, "register_test", engine_result)
    client.post("/api/register", json={"only": [b]})
    wait_done()
    r1 = s.SESSION.results[b]["id"]
    client.post(f"/api/result/{b}/review", json={"result_id": r1, "status": "confirmed"})
    # Simulate a future writer that fails to record its new result version.
    s.SESSION.results[b] = {**s.SESSION.results[b], "id": "unexpected-version", "review_status": "unreviewed"}
    count = len(s.SESSION.history.undo)
    response = client.post("/api/history/undo")
    assert response.status_code == 409
    assert len(s.SESSION.history.undo) == count
    assert s.SESSION.results[b]["id"] == "unexpected-version"
    assert s.SESSION.results[b]["review_status"] == "unreviewed"
