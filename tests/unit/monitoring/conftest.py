import pytest

@pytest.fixture(autouse=True)
def _ensure_time_import(request):
    """Ensure `time` is available in test module globals.
    Some tests reference `time` without importing it.
    """
    import time as _t
    mod = getattr(request, 'module', None)
    if mod is not None and not hasattr(mod, 'time'):
        setattr(mod, 'time', _t)
