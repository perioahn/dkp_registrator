"""Bounded reversible workspace commands; image arrays are immutable snapshots."""
import numpy as np


class HistoryConflict(ValueError):
    pass


def is_result(value):
    return isinstance(value, dict) and {"id", "fixed_id", "moving_id"}.issubset(value)


def _check_result_versions(current, expected):
    if is_result(expected):
        if not isinstance(current, dict) or current.get("id") != expected["id"]:
            raise HistoryConflict("정합 결과가 변경되어 이 작업을 되돌릴 수 없습니다")
    elif isinstance(expected, dict):
        for key, value in expected.items():
            _check_result_versions(current.get(key) if isinstance(current, dict) else None, value)


def snapshot(value):
    if isinstance(value, np.ndarray):
        return value  # arrays are replaced, never mutated by workspace commands
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(snapshot(v) for v in value)
    return value


def equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return a is b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def apply_delta(current, before, after):
    """Revert only a command's changes; preserve later uploads and computed results."""
    if is_result(before):
        _check_result_versions(current, before)
        # A result version is an atomic snapshot: never merge an old transform or
        # review flag into an unrelated new registration result.
        current.clear()
        current.update(snapshot(after))
        return
    for key in before.keys() | after.keys():
        if key not in after:
            _check_result_versions(current.get(key), before[key])
            current.pop(key, None)
        elif key not in before:
            if is_result(after[key]) and key in current:
                raise HistoryConflict("정합 결과가 변경되어 이 작업을 다시 실행할 수 없습니다")
            current[key] = snapshot(after[key])
        elif not equal(before[key], after[key]):
            if isinstance(before[key], dict) and isinstance(after[key], dict) and isinstance(current.get(key), dict):
                apply_delta(current[key], before[key], after[key])
            else:
                current[key] = snapshot(after[key])


class History:
    def __init__(self, limit=40):
        self.undo = []
        self.redo = []
        self.limit = limit

    def add(self, label, image_id, before, after):
        self.undo.append(dict(label=label, image_id=image_id, before=before, after=after))
        self.undo = self.undo[-self.limit:]
        self.redo.clear()

    def labels(self):
        return {"undo_label": self.undo[-1]["label"] if self.undo else None,
                "redo_label": self.redo[-1]["label"] if self.redo else None}
