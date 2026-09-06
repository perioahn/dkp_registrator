"""Reference-aware session state, isolated paths and revisioned edit history."""
import os
import threading
import uuid
from webapp.history import History, snapshot, apply_delta


class Session:
    def __init__(self, root):
        # Never remove the previous session on import/startup.
        self.dir = os.path.join(root, "sessions", uuid.uuid4().hex)
        os.makedirs(self.dir, exist_ok=True)
        self.images = {}
        self.order = []
        self.masks = {}
        self.anchors = {}
        self.result_pairs = {}
        self.fixed = None
        self.revision = 0
        self.running = False
        self.job = None
        self.pending_fixed = None
        self.lock = threading.RLock()
        self.history = History()

    def fixed_id(self):
        return self.fixed

    def moving_ids(self):
        return [i for i in self.order if i != self.fixed]

    @property
    def results(self):
        return self.result_pairs.setdefault(self.fixed, {})

    def set_fixed(self, image_id):
        self.fixed = image_id
        for i, im in self.images.items():
            im["role"] = "fixed" if i == image_id else "moving"

    def snapshot(self):
        return snapshot({k: getattr(self, k) for k in
                         ("images", "order", "masks", "anchors", "result_pairs", "fixed")})

    def record(self, label, image_id, before):
        self.revision += 1
        self.history.add(label, image_id, before, self.snapshot())

    def restore(self, before, after):
        current = self.snapshot()
        added_later = [i for i in current["order"] if i not in before["order"] and i not in after["order"]]
        apply_delta(current, before, after)
        current["order"] += [i for i in added_later if i not in current["order"]]
        for k, v in current.items():
            setattr(self, k, v)
        self.set_fixed(self.fixed)
        self.revision += 1
