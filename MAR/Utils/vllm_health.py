"""Periodic vLLM health checker for long-running training jobs.

Usage:
    from MAR.Utils.vllm_health import VllmHealthChecker

    checker = VllmHealthChecker.from_profile(interval_seconds=3600)
    # In training loop:
    checker.check_or_exit()  # exits if servers are dead
"""

import sys
import time
import requests
from typing import Dict, Optional
from loguru import logger


class VllmHealthChecker:
    """Periodically probes vLLM model servers and exits if any are unhealthy."""

    def __init__(
        self,
        model_urls: Dict[str, str],
        interval_seconds: float = 3600.0,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Args:
            model_urls: {model_name: base_url} e.g. {"deepseek-ai/...": "http://127.0.0.1:8001/v1"}
            interval_seconds: Minimum seconds between health checks (default 1 hour).
            timeout_seconds: HTTP timeout per probe request.
            max_retries: Retry count before declaring a model unhealthy.
            retry_delay: Seconds between retries.
        """
        self.model_urls = model_urls
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._last_check_time: float = time.monotonic()

    @classmethod
    def from_profile(cls, interval_seconds: float = 3600.0, **kwargs) -> "VllmHealthChecker":
        """Create a checker from the llm_profile_full.json model_base_urls."""
        from MAR.LLM.llm_profile_full import model_base_urls
        if not model_base_urls:
            logger.warning("No model_base_urls found in profile; health checker disabled.")
            return cls({}, interval_seconds=interval_seconds, **kwargs)
        return cls(model_base_urls, interval_seconds=interval_seconds, **kwargs)

    def _probe_model(self, name: str, base_url: str) -> Optional[str]:
        """Probe a single model. Returns error string or None if healthy."""
        models_url = f"{base_url.rstrip('/')}/models"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(models_url, timeout=self.timeout_seconds)
                if resp.status_code == 200:
                    return None  # healthy
                err = f"HTTP {resp.status_code}"
            except requests.ConnectionError:
                err = "connection refused"
            except requests.Timeout:
                err = "timeout"
            except Exception as exc:
                err = str(exc)

            if attempt < self.max_retries:
                logger.warning(
                    "[HealthCheck] {} unreachable (attempt {}/{}): {} — retrying in {}s",
                    name, attempt, self.max_retries, err, self.retry_delay,
                )
                time.sleep(self.retry_delay)
        return err

    def check_now(self) -> bool:
        """Run health check immediately. Returns True if all healthy, False otherwise."""
        if not self.model_urls:
            return True

        logger.info("[HealthCheck] Probing {} vLLM servers...", len(self.model_urls))
        unhealthy = {}
        for name, base_url in self.model_urls.items():
            err = self._probe_model(name, base_url)
            if err:
                unhealthy[name] = err

        self._last_check_time = time.monotonic()

        if unhealthy:
            for name, err in unhealthy.items():
                logger.error("[HealthCheck] UNHEALTHY: {} — {}", name, err)
            return False

        logger.info("[HealthCheck] All {} models healthy.", len(self.model_urls))
        return True

    def check_or_exit(self) -> None:
        """Check if interval has elapsed; if so, probe servers and exit on failure."""
        elapsed = time.monotonic() - self._last_check_time
        if elapsed < self.interval_seconds:
            return

        if not self.check_now():
            logger.critical(
                "[HealthCheck] vLLM servers are unhealthy after {} retries. "
                "Exiting to avoid wasting compute. "
                "Check vLLM logs and restart the job.",
                self.max_retries,
            )
            sys.exit(1)
