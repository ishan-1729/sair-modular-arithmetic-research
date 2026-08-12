from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def idempotent_torch_interop_thread_setting() -> Iterator[None]:
    """Make repeated requests for the active PyTorch interop count harmless.

    PyTorch permits setting this process-global value only once. Several frozen
    experiment tests independently request the same value, so their result must
    not depend on collection order. Requests for a different value still reach
    PyTorch and fail closed.
    """

    original = torch.set_num_interop_threads

    def set_num_interop_threads(threads: int) -> None:
        if torch.get_num_interop_threads() == threads:
            return
        original(threads)

    torch.set_num_interop_threads = set_num_interop_threads
    try:
        yield
    finally:
        torch.set_num_interop_threads = original
