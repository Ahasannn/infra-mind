"""
Blackbox metrics provider — artificial congestion signals for API models.

Provides the same metric interface as ``metrics_watcher`` (queue_depth,
e2e_avg, kv_cache_usage) so the InfraMind executor sees a consistent
state vector regardless of whitebox vs blackbox models.

All metrics are computed over a 60-second sliding window:
  - congestion  = requests_in_window / rpm_limit   (∈ [0, 2+])
  - e2e_avg     = mean(latencies_in_window)         (client-side, seconds)
  - kv_cache    = 0.0                               (no GPU memory for blackbox)
"""

import threading
import time
from collections import defaultdict, deque
from typing import Dict, Optional

from loguru import logger

# Shared metrics dict — same structure as metrics_watcher.model_metrics
# so the router can read from the same global dict.
from MAR.InfraMind.metrics_watcher import model_metrics


class BlackboxMetricsProvider:
    """Track RPM usage and client-side latency for blackbox API models.

    Call ``record_request()`` after each API call to update the sliding window.
    The provider writes into the shared ``model_metrics`` dict so the router's
    ``_get_raw_system_metrics()`` works without modification.

    All metrics use the same 60-second sliding window for consistency
    (congestion, e2e_avg). This matches vLLM's windowed metric reporting.
    """

    def __init__(
        self,
        rpm_limits: Dict[str, int],
        window_seconds: float = 60.0,
    ) -> None:
        self._rpm_limits = rpm_limits
        self._window = window_seconds
        self._lock = threading.Lock()

        # Sliding window of (timestamp, latency) per model
        self._requests: Dict[str, deque] = defaultdict(deque)

        # Initialize metrics for all models
        for model_name in rpm_limits:
            model_metrics[model_name] = {
                "num_requests_running": 0.0,
                "num_requests_waiting": 0.0,
                "kv_cache_usage_perc": 0.0,
                "e2e_avg": 0.0,
                "ttft_avg": 0.0,
                "itl_avg": 0.0,
                "queue_avg": 0.0,
                "inference_avg": 0.0,
            }
        logger.info(
            "[BlackboxMetrics] Initialized for {} models (window={}s)",
            len(rpm_limits), window_seconds,
        )

    def record_request(
        self,
        model_name: str,
        latency_seconds: float,
        ttft_seconds: Optional[float] = None,
    ) -> None:
        """Record a completed API request and update metrics."""
        now = time.monotonic()
        with self._lock:
            self._requests[model_name].append((now, latency_seconds))
            self._prune(model_name, now)
            self._update_model_metrics(model_name, now)

    def get_congestion(self, model_name: str) -> float:
        """Current congestion level for a model (requests_in_window / rpm_limit)."""
        now = time.monotonic()
        with self._lock:
            self._prune(model_name, now)
            count = len(self._requests[model_name])
        rpm_limit = self._rpm_limits.get(model_name, 60)
        return count / max(rpm_limit, 1)

    def _prune(self, model_name: str, now: float) -> None:
        """Remove requests outside the sliding window."""
        q = self._requests[model_name]
        cutoff = now - self._window
        while q and q[0][0] < cutoff:
            q.popleft()

    def _windowed_e2e_avg(self, model_name: str) -> float:
        """Mean E2E latency over the current sliding window."""
        q = self._requests[model_name]
        if not q:
            return 0.0
        return sum(lat for _, lat in q) / len(q)

    def _update_model_metrics(self, model_name: str, now: float) -> None:
        """Write metrics in the same format as metrics_watcher."""
        count = len(self._requests[model_name])
        rpm_limit = self._rpm_limits.get(model_name, 60)
        congestion = count / max(rpm_limit, 1)
        e2e = self._windowed_e2e_avg(model_name)

        # Update only latency-related fields — leave num_requests_running/waiting
        # to BlackboxRequestQueue which tracks real in-flight counts.
        existing = model_metrics.get(model_name, {})
        existing["kv_cache_usage_perc"] = 0.0
        existing["e2e_avg"] = e2e
        existing["ttft_avg"] = 0.0
        existing["itl_avg"] = 0.0
        existing["queue_avg"] = 0.0
        existing["inference_avg"] = 0.0
        existing["congestion"] = congestion
        model_metrics[model_name] = existing
