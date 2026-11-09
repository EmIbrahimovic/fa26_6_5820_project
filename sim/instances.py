"""Prefill and decode instances (plan §3.6, §3.8).

PrefillInstance
  FIFO queue. When idle it takes queued prompts up to ``max_batch_tokens``
  (always at least one) and runs them as one batch whose duration comes from
  the profile. All prompts in the batch finish together; their KV caches are
  produced uniformly over the batch (layer by layer).

  The prefill side's KV memory is not modelled (assumed large enough to hold
  KV until it has been sent).

DecodeInstance
  KV memory measured in tokens. A request must *reserve* memory for
  ``prompt + output`` tokens before its KV may be sent (KV needs somewhere to
  land); reservations are granted FIFO. Once the KV has fully arrived the
  request joins the running batch at the next iteration boundary (continuous
  batching). Each iteration produces one token for every request in the batch.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import Simulation
    from .workload import Request


class PrefillInstance:
    def __init__(self, idx: int, inst: int, sim: "Simulation", max_batch_tokens: int):
        self.idx = idx
        self.inst = inst
        self.sim = sim
        self.max_batch_tokens = max_batch_tokens
        self.queue: deque["Request"] = deque()
        self.queued_tokens = 0
        self.busy = False
        self.busy_until = 0.0
        self.busy_time = 0.0

    def est_wait(self, now: float) -> float:
        """Rough time until a newly queued prompt would start."""
        run = max(self.busy_until - now, 0.0) if self.busy else 0.0
        return run + (self.sim.perf.prefill_time(self.queued_tokens) if self.queued_tokens else 0.0)

    def enqueue(self, req: "Request") -> None:
        self.queue.append(req)
        self.queued_tokens += req.prompt_tokens
        if not self.busy:
            self._start()

    def _start(self) -> None:
        batch = [self.queue.popleft()]
        tokens = batch[0].prompt_tokens
        while self.queue and tokens + self.queue[0].prompt_tokens <= self.max_batch_tokens:
            r = self.queue.popleft()
            batch.append(r)
            tokens += r.prompt_tokens
        self.queued_tokens -= tokens
        sim = self.sim
        now = sim.engine.now
        t = sim.perf.prefill_time_batch([r.prompt_tokens for r in batch])
        self.busy = True
        self.busy_until = now + t
        self.busy_time += t
        for r in batch:
            r.prefill_start = now
            r.prefill_end = now + t
            r.prefill_time = t
        for r in batch:
            sim.on_prefill_start(r)
        sim.engine.at(now + t, self._finish, batch)

    def _finish(self, batch: list["Request"]) -> None:
        self.busy = False
        for r in batch:
            self.sim.on_prefill_done(r)
        if self.queue and not self.busy:
            self._start()


class DecodeInstance:
    def __init__(self, idx: int, inst: int, sim: "Simulation", capacity_tokens: int, max_batch: int):
        self.idx = idx
        self.inst = inst
        self.sim = sim
        self.capacity = capacity_tokens
        self.max_batch = max_batch
        self.reserved = 0           # tokens reserved (bound + transferring + running)
        self.waiting: deque["Request"] = deque()  # bound, waiting for memory
        self.waiting_tokens = 0
        self.inbound = 0            # reserved but KV not yet fully arrived
        self.ready: deque["Request"] = deque()    # KV arrived, waiting for next iteration
        self.batch: list["Request"] = []
        self.batch_ctx = 0          # sum of context lengths in the running batch
        self.running = False
        self.busy_time = 0.0

    @property
    def free_tokens(self) -> int:
        return self.capacity - self.reserved - self.waiting_tokens

    def request_memory(self, req: "Request") -> bool:
        """Reserve KV memory now if possible, else queue (FIFO)."""
        need = req.mem_tokens
        if not self.waiting and self.reserved + need <= self.capacity:
            self.reserved += need
            self.inbound += 1
            return True
        self.waiting.append(req)
        self.waiting_tokens += need
        return False

    def _release(self, req: "Request") -> None:
        self.reserved -= req.mem_tokens
        while self.waiting and self.reserved + self.waiting[0].mem_tokens <= self.capacity:
            r = self.waiting.popleft()
            self.waiting_tokens -= r.mem_tokens
            self.reserved += r.mem_tokens
            self.inbound += 1
            self.sim.on_memory_reserved(r)

    def kv_arrived(self, req: "Request") -> None:
        self.inbound -= 1
        self.ready.append(req)
        if not self.running:
            self._iterate()

    def _iterate(self) -> None:
        while self.ready and len(self.batch) < self.max_batch:
            r = self.ready.popleft()
            self.batch.append(r)
            self.batch_ctx += r.prompt_tokens + 1
        if not self.batch:
            self.running = False
            return
        self.running = True
        t = self.sim.perf.decode_iter_time(len(self.batch), self.batch_ctx)
        self.busy_time += t
        self.sim.engine.after(t, self._iter_done)

    def _iter_done(self) -> None:
        now = self.sim.engine.now
        keep = []
        finished = []
        for r in self.batch:
            r.decoded += 1
            self.batch_ctx += 1
            if r.decoded == 1:
                r.second_token = now
            if 1 + r.decoded >= r.output_tokens:
                finished.append(r)
            else:
                keep.append(r)
        self.batch = keep
        for r in finished:
            r.completion = now
            self.batch_ctx -= r.prompt_tokens + 1 + r.decoded
            self._release(r)
        self._iterate()
