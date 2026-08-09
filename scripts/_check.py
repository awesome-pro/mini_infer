"""Minimal check harness: plain asserts, no third-party test runner.

Usage::

    python scripts/check_engine.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_FAILURES: list[str] = []
_PASSED = 0


def check(description: str, fn: Callable[[], Any]) -> None:
    """Run one check, recording failure instead of aborting the run."""
    global _PASSED
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - report everything, keep going
        _FAILURES.append(f"{description}: {type(exc).__name__}: {exc}")
    else:
        _PASSED += 1


def close(actual: float, expected: float, *, rel: float = 1e-9, abs_: float = 1e-12) -> None:
    """Assert two floats agree within a relative tolerance."""
    if abs(actual - expected) > max(abs_, rel * max(abs(actual), abs(expected))):
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def expect_raises(exc_type: type[BaseException], match: str, fn: Callable[[], Any]) -> None:
    try:
        fn()
    except exc_type as exc:
        assert match in str(exc), f"expected {match!r} in {exc!r}"
        return
    raise AssertionError(f"expected {exc_type.__name__} matching {match!r}")


def report(title: str) -> int:
    """Print the summary and return a process exit code."""
    total = _PASSED + len(_FAILURES)
    if _FAILURES:
        print(f"\n{title}: {_PASSED}/{total} checks passed")
        for failure in _FAILURES:
            print(f"  FAIL  {failure}")
        return 1
    print(f"\n{title}: all {total} checks passed")
    return 0
