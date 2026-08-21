"""Minimal assert-based test harness (the project has no pytest).

Each test function takes a single `Check` argument and uses c.eq / c.true /
c.raises. `run_all` discovers `test_*` functions in a module's globals, runs
each, and prints a PASS/FAIL summary, exiting non-zero on any failure.
"""
from __future__ import annotations

import traceback
from typing import Callable


class CheckFailure(AssertionError):
    pass


class Check:
    """Collects assertions for one test; raises on first failure."""

    def eq(self, got, want, msg: str = ""):
        if got != want:
            raise CheckFailure(f"{msg}: got {got!r}, want {want!r}")

    def true(self, cond: bool, msg: str = ""):
        if not cond:
            raise CheckFailure(f"expected true: {msg}")

    def raises(self, fn: Callable, exc: type[BaseException], msg: str = ""):
        try:
            fn()
        except exc:
            return
        except BaseException as e:  # wrong exception type
            raise CheckFailure(f"{msg}: expected {exc.__name__}, got {type(e).__name__}: {e}")
        raise CheckFailure(f"{msg}: expected {exc.__name__}, nothing raised")


def run_all(module_name: str, ns: dict) -> None:
    import sys

    tests = sorted((k, v) for k, v in ns.items()
                   if k.startswith("test_") and callable(v))
    passed, failed = 0, 0
    print(f"\n=== {module_name}: {len(tests)} tests ===")
    for name, fn in tests:
        try:
            fn(Check())
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"  ok    {name}")
    print(f"--- {passed} passed, {failed} failed ---")
    if failed:
        sys.exit(1)
