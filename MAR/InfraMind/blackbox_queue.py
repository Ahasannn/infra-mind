"""
Local request queue with per-model RPM throttling and EDF scheduling.

Each model has an RPM (requests per minute) limit — the blackbox equivalent
of vLLM's ``max_num_seqs``.  When a model's RPM is exhausted, new requests
queue locally and are dispatched in earliest-deadline-first (EDF) order.
Tighter budgets get served first, just like vLLM's priority scheduling.

The queue updates ``model_metrics`` in real time so the InfraMind executor
sees genuine congestion signals (running count, waiting count) and can
route away from overloaded models.

Usage (called once at startup by entry point):

    queue = BlackboxRequestQueue.configure(
        rpm_limits={"gemini-2.0-flash": 10, ...},
        metrics_provider=blackbox_metrics,
    )

LLM calls are automatically routed through the queue via the
``blackbox_setup.py`` wrapper on ``_call_llm_stream``.
"""

import heapq
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

from loguru import logger

from MAR.InfraMind.metrics_watcher import model_metrics


# ---------------------------------------------------------------------------
# Per-model RPM gate with EDF priority
# ---------------------------------------------------------------------------

class _ModelGate:
    """RPM-based throttle with priority queuing (earliest deadline first).

    Tracks completed requests in a 60-second sliding window.  When the
    window is full (count >= rpm_limit), new requests wait in a min-heap
    ordered by deadline.  As old requests age out of the window, the
    highest-priority waiter is dispatched.

    Also tracks in-flight (running) requests separately from completed ones
    so the executor sees both running and waiting counts.
    """

    def __init__(self, model_name: str, rpm_limit: int) -> None:
        self.model_name = model_name
        self.rpm_limit = rpm_limit
        self._lock = threading.Lock()
        self._timestamps: deque = deque()  # monotonic times of dispatched requests
        self._running = 0                  # currently in-flight
        self._heap: list = []              # min-heap of (deadline, tiebreaker, Event)
        self._counter = 0                  # monotonic tiebreaker
        self._drain_timer: Optional[threading.Timer] = None

    def acquire(self, deadline: float) -> float:
        """Block until an RPM slot is available.  Returns time spent waiting."""
        wait_start = time.perf_counter()

        with self._lock:
            self._prune()
            if len(self._timestamps) < self.rpm_limit:
                # Slot available — dispatch immediately
                self._timestamps.append(time.monotonic())
                self._running += 1
                self._update_metrics()
                return 0.0
            # RPM exhausted — queue with EDF priority
            event = threading.Event()
            self._counter += 1
            heapq.heappush(self._heap, (deadline, self._counter, event))
            self._update_metrics()
            self._schedule_drain()

        # Wait outside the lock
        event.wait()
        wait_time = time.perf_counter() - wait_start
        return wait_time

    def release(self) -> None:
        """Mark a request as completed.  Dispatches next waiter if RPM allows."""
        with self._lock:
            self._running = max(0, self._running - 1)
            self._try_dispatch()
            self._update_metrics()

    def _prune(self) -> None:
        """Remove timestamps older than 60 seconds from the sliding window."""
        cutoff = time.monotonic() - 60.0
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def _try_dispatch(self) -> None:
        """Dispatch queued waiters if RPM slots have freed up (must hold lock)."""
        self._prune()
        while self._heap and len(self._timestamps) < self.rpm_limit:
            _, _, event = heapq.heappop(self._heap)
            self._timestamps.append(time.monotonic())
            self._running += 1
            event.set()
        # If there are still waiters, schedule a timer for when the oldest
        # request ages out of the window
        if self._heap:
            self._schedule_drain()

    def _schedule_drain(self) -> None:
        """Schedule a timer to try dispatching when the oldest request expires."""
        if self._drain_timer is not None:
            self._drain_timer.cancel()
        if not self._timestamps or not self._heap:
            self._drain_timer = None
            return
        oldest = self._timestamps[0]
        wait = max(0.01, (oldest + 60.0) - time.monotonic())
        self._drain_timer = threading.Timer(wait, self._on_drain_timer)
        self._drain_timer.daemon = True
        self._drain_timer.start()

    def _on_drain_timer(self) -> None:
        """Called when the oldest request ages out — try to dispatch waiters."""
        with self._lock:
            self._drain_timer = None
            self._try_dispatch()
            self._update_metrics()

    def _update_metrics(self) -> None:
        """Write real running/waiting counts into shared model_metrics."""
        model_metrics[self.model_name] = {
            **model_metrics.get(self.model_name, {}),
            "num_requests_running": float(self._running),
            "num_requests_waiting": float(len(self._heap)),
        }

    @property
    def running(self) -> int:
        with self._lock:
            return self._running

    @property
    def waiting(self) -> int:
        with self._lock:
            return len(self._heap)

    @property
    def window_count(self) -> int:
        """Requests dispatched in the current 60s window."""
        with self._lock:
            self._prune()
            return len(self._timestamps)


# ---------------------------------------------------------------------------
# Queue manager
# ---------------------------------------------------------------------------

class BlackboxRequestQueue:
    """Per-model local queue with RPM throttling and EDF scheduling.

    This is the blackbox equivalent of vLLM's request scheduling:
    - RPM limit acts like max_num_seqs (controls throughput)
    - EDF priority ensures urgent requests (tight budgets) go first
    - model_metrics updated in real time for executor routing decisions

    When RPM is exceeded, requests queue locally.  As old requests age
    out of the 60-second sliding window, queued requests dispatch in
    earliest-deadline-first order.
    """

    _instance: Optional["BlackboxRequestQueue"] = None

    def __init__(
        self,
        rpm_limits: Dict[str, int],
        metrics_provider: Optional[Any] = None,
    ) -> None:
        self._gates: Dict[str, _ModelGate] = {
            model: _ModelGate(model, rpm)
            for model, rpm in rpm_limits.items()
        }
        self._metrics_provider = metrics_provider
        self._total_wait = 0.0
        self._total_requests = 0
        self._lock = threading.Lock()
        logger.info(
            "[BlackboxQueue] RPM limits: {}",
            {m: r for m, r in rpm_limits.items()},
        )

    @classmethod
    def configure(
        cls,
        rpm_limits: Dict[str, int],
        metrics_provider: Optional[Any] = None,
    ) -> "BlackboxRequestQueue":
        """Create and register the global queue instance."""
        cls._instance = cls(rpm_limits, metrics_provider)
        return cls._instance

    @classmethod
    def instance(cls) -> Optional["BlackboxRequestQueue"]:
        return cls._instance

    def submit(
        self,
        model_name: str,
        call_fn: Callable[[], Any],
        deadline: float = float("inf"),
    ) -> Any:
        """Submit a request through the queue.

        Blocks until an RPM slot is available (EDF priority).
        After execution, records latency in the metrics provider.

        Args:
            model_name: Which model this request targets.
            call_fn: Zero-arg callable that makes the actual API call.
            deadline: arrival_time + budget_total (smaller = higher priority).
        """
        gate = self._gates.get(model_name)
        if gate is None:
            return call_fn()

        # Acquire RPM slot (may block if RPM exhausted)
        wait_time = gate.acquire(deadline)
        with self._lock:
            self._total_wait += wait_time
            self._total_requests += 1

        start = time.perf_counter()
        try:
            result = call_fn()
            return result
        finally:
            latency = time.perf_counter() - start
            gate.release()
            if self._metrics_provider is not None:
                self._metrics_provider.record_request(model_name, latency)

    def get_status(self) -> Dict[str, Dict[str, int]]:
        """Return {model: {running, waiting, rpm_limit, window_count}}."""
        return {
            model: {
                "running": gate.running,
                "waiting": gate.waiting,
                "rpm_limit": gate.rpm_limit,
                "window_count": gate.window_count,
            }
            for model, gate in self._gates.items()
        }

    def is_idle(self) -> bool:
        """True when all gates have 0 running and 0 waiting."""
        return all(
            gate.running == 0 and gate.waiting == 0
            for gate in self._gates.values()
        )

    def wait_until_idle(self, poll_interval: float = 2.0, timeout: float = 300.0) -> bool:
        """Block until all model gates are idle (0 running, 0 waiting).

        Returns True if idle, False if timed out.
        """
        if self.is_idle():
            return True
        logger.info("[BlackboxQueue] Waiting for queues to drain...")
        start = time.perf_counter()
        while not self.is_idle():
            elapsed = time.perf_counter() - start
            if elapsed > timeout:
                logger.warning("[BlackboxQueue] Drain timeout after {:.0f}s", elapsed)
                return False
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                status = self.get_status()
                busy = {m: s for m, s in status.items() if s["running"] > 0 or s["waiting"] > 0}
                logger.info("[BlackboxQueue] Still draining ({:.0f}s): {}", elapsed, busy)
            time.sleep(poll_interval)
        logger.info("[BlackboxQueue] All queues drained.")
        return True

    def log_stats(self) -> None:
        avg_wait = self._total_wait / max(self._total_requests, 1)
        logger.info(
            "[BlackboxQueue] {} total requests | avg queue wait: {:.3f}s",
            self._total_requests, avg_wait,
        )
        for model, gate in self._gates.items():
            logger.info(
                "[BlackboxQueue]   {} — running={}, waiting={}, window={}/{}rpm",
                model, gate.running, gate.waiting,
                gate.window_count, gate.rpm_limit,
            )
