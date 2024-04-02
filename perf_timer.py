"""Functionality to help with timing performance critical paths.

This can be run as a service to autostart so it is always available to take
pictures through the connected camera.
"""

import time
from absl import logging as log


class PerformanceTimer:
    """Non-implementation by default for PROD mode."""

    def __init__(self) -> None:
        pass

    def start(self, name: str) -> None:
        pass

    def end(self, name: str) -> None:
        pass

    def print_report(self) -> None:
        pass

    def quick_start(self) -> None:
        pass

    def quick_end(self) -> None:
        pass


class PerformanceTimerImpl(PerformanceTimer):
    """Default implementation for DEV mode."""

    def __init__(self) -> None:
        self._starts = {}
        self._ends = {}
        self._tic = 0.0

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter_ns()

    def end(self, name: str) -> None:
        self._ends[name] = time.perf_counter_ns()
        tic = self._starts[name]
        toc = self._ends[name]
        log.info(f"{name} took {(toc - tic)/1000000 :.3f}ms")

    def print_report(self) -> None:
        log.info("===== Performance Report =====")
        for id in self._starts:
            start = self._starts[id]
            if id not in self._ends:
                log.error(f'Section "{id}" never ended.')
                continue
            end = self._ends[id]

            duration_millis = (end - start) // 1_000_000
            log.info(f"{id}\ttook {duration_millis} ms")

    def quick_start(self) -> None:
        self._tic: float = time.perf_counter()

    def quick_end(self, name) -> None:
        tic = self._tic
        toc: float = time.perf_counter()
        log.info(f"{name} took {1000*(toc - tic):.3f}ms")
