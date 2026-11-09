"""Discrete-event engine (plan §3.1).

A priority queue of (time, seq, callback). ``run`` pops the earliest event,
advances ``now`` and calls the callback, which may schedule more events.

Events can be marked ``daemon``: periodic housekeeping (router snapshot
refresh, background traffic) that should not keep the simulation alive on its
own. The run ends when only daemon events remain.
"""
from __future__ import annotations

import heapq
import itertools
from typing import Any, Callable


class Event:
    __slots__ = ("time", "fn", "args", "daemon", "cancelled")

    def __init__(self, time: float, fn: Callable, args: tuple, daemon: bool):
        self.time = time
        self.fn = fn
        self.args = args
        self.daemon = daemon
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class Engine:
    def __init__(self) -> None:
        self.now = 0.0
        self._heap: list[tuple[float, int, Event]] = []
        self._seq = itertools.count()
        self._live = 0  # number of queued non-daemon events
        self.events_processed = 0

    def at(self, time: float, fn: Callable, *args: Any, daemon: bool = False) -> Event:
        if time < self.now:
            # Floating-point slop: never schedule into the past.
            time = self.now
        ev = Event(time, fn, args, daemon)
        heapq.heappush(self._heap, (time, next(self._seq), ev))
        if not daemon:
            self._live += 1
        return ev

    def after(self, delay: float, fn: Callable, *args: Any, daemon: bool = False) -> Event:
        return self.at(self.now + delay, fn, *args, daemon=daemon)

    def run(self, until: float | None = None) -> None:
        heap = self._heap
        while heap and self._live > 0:
            t, _, ev = heap[0]
            if until is not None and t > until:
                break
            heapq.heappop(heap)
            if not ev.daemon:
                self._live -= 1
            if ev.cancelled:
                continue
            self.now = t
            self.events_processed += 1
            ev.fn(*ev.args)
