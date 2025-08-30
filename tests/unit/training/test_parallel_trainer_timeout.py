import time
import pytest
from adan_trading_bot.utils.timeout_manager import TimeoutManager, TimeoutException


def test_parallel_trainer_timeout_context_manager():
    # This test validates that the timeout context used by the parallel trainer works.
    # It does not run the heavy trainer; it focuses on TimeoutManager behavior in this context.
    t0 = time.time()
    with pytest.raises(TimeoutException):
        with TimeoutManager(timeout=0.2).limit():
            time.sleep(0.6)
    assert time.time() - t0 < 1.5  # should exit roughly around timeout
