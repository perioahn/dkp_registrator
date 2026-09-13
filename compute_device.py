"""Process-local compute selection for repeatable CPU/accelerator testing."""
_mode = 'auto'


def accelerator():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return None


def current():
    return 'cpu' if _mode == 'cpu' else (accelerator() or 'cpu')


def mode():
    return _mode


def select(value):
    global _mode
    if value not in ('auto', 'cpu'):
        raise ValueError('지원하지 않는 실행 장치입니다.')
    if value == 'auto' and accelerator() is None:
        raise ValueError('현재 실행 환경에서 사용할 수 있는 GPU 가속이 없습니다. 아래 CPU 안내를 확인하세요.')
    _mode = value
