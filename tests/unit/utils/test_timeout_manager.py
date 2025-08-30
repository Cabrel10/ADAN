import time
import pytest

from adan_trading_bot.utils.timeout_manager import (
    TimeoutManager,
    TimeoutException,
)


def test_timeout_manager_success_no_timeout():
    manager = TimeoutManager(timeout=1.5)
    start = time.time()
    with manager.limit():
        time.sleep(0.2)
    assert time.time() - start < 1.5


def test_timeout_manager_triggered():
    manager = TimeoutManager(timeout=0.2)
    with pytest.raises(TimeoutException):
        with manager.limit():
            time.sleep(0.6)


def test_timeout_manager_decorator():
    manager = TimeoutManager(timeout=0.2)

    @manager.decorator
    def long_job():
        time.sleep(0.6)

    with pytest.raises(TimeoutException):
        long_job()


def test_timeout_manager_cleanup_called():
    called = {"v": False}

    def cleanup():
        called["v"] = True

    manager = TimeoutManager(timeout=0.2, cleanup_callback=cleanup)
    with pytest.raises(TimeoutException):
        with manager.limit():
            time.sleep(0.6)
    assert called["v"] is True


def test_timeout_manager_very_short_timeout():
    # Use a very small timeout and ensure it triggers reliably
    manager = TimeoutManager(timeout=0.03)
    with pytest.raises(TimeoutException):
        with manager.limit():
            time.sleep(0.2)
