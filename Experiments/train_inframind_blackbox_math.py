"""
InfraMind training with blackbox (API) model pool on MATH dataset.

Sets up the blackbox environment (API keys, model pool, metrics provider,
response cache) then delegates to the shared InfraMind training loop.

Usage:
    python Experiments/train_inframind_blackbox_math.py \
        --limit 100 --epochs 5 --budget-tiers 30,100,300

API keys should be set in ``.env`` (GEMINI_API_KEY, OPENROUTER_API_KEY).
Response cache persists in ``cache/blackbox_math_inframind.db`` by default.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load .env BEFORE any other imports
from dotenv import load_dotenv
load_dotenv()

from loguru import logger

from MAR.LLM.model_pool import ModelPool
from MAR.LLM.blackbox_setup import configure_blackbox_env
from MAR.LLM.response_cache import ResponseCache, enable_cache
from MAR.InfraMind.blackbox_metrics import BlackboxMetricsProvider

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "..", "MAR", "LLM", "llm_profile_blackbox.json")
DEFAULT_CACHE_PATH = "cache/blackbox_math_inframind.db"


def _setup_blackbox_pool(profile_path: str) -> ModelPool:
    """Load blackbox pool and configure environment."""
    pool = ModelPool(profile_path)
    configure_blackbox_env(pool)
    return pool


def _patch_inframind_for_blackbox(pool: ModelPool) -> None:
    """Patch InfraMind modules to use blackbox models instead of vLLM.

    - Overrides _load_models_from_profile() to return blackbox model names
    - Patches InfraMindEnv to use BlackboxMetricsProvider (no vLLM polling)
    - Disables vLLM health checks
    """
    import MAR.InfraMind.inframind_router as router_mod

    # 1. Override model loading to return blackbox models
    router_mod._load_models_from_profile = lambda: pool.models
    logger.info("[Blackbox] Patched _load_models_from_profile → {}", pool.models)

    # 2. Patch InfraMindEnv to skip vLLM metrics watcher
    import MAR.InfraMind.env as env_mod
    _original_env_init = env_mod.InfraMindEnv.__init__

    # Create blackbox metrics provider
    blackbox_provider = BlackboxMetricsProvider(rpm_limits=pool.get_rpm_limits())

    # Local request queue — RPM throttling + EDF scheduling
    from MAR.InfraMind.blackbox_queue import BlackboxRequestQueue
    BlackboxRequestQueue.configure(
        rpm_limits=pool.get_rpm_limits(),
        metrics_provider=blackbox_provider,
    )

    def _env_init_blackbox(self, *args, **kwargs):
        # Pass empty metrics_url_map to prevent vLLM polling
        kwargs["metrics_url_map"] = {}
        _original_env_init(self, *args, **kwargs)
        # Store provider reference for potential use
        self._blackbox_metrics = blackbox_provider
        logger.info("[Blackbox] InfraMindEnv initialized with blackbox metrics (no vLLM polling)")

    env_mod.InfraMindEnv.__init__ = _env_init_blackbox

    # 3. Disable vLLM health checker
    import MAR.Utils.vllm_health as health_mod
    if hasattr(health_mod, "VllmHealthChecker"):
        class _NoOpHealthChecker:
            """Dummy health checker that always passes."""
            @classmethod
            def from_profile(cls, **kwargs):
                return cls()
            def check_now(self):
                return True
            def check_or_exit(self):
                pass
        health_mod.VllmHealthChecker = _NoOpHealthChecker

    # 4. Replace vLLM drain with blackbox queue drain
    import MAR.InfraMind.training as training_mod

    def _blackbox_drain(poll_interval: float = 5.0) -> bool:
        """Wait for all blackbox RPM queues to clear before next sweep."""
        queue = BlackboxRequestQueue.instance()
        if queue is not None:
            return queue.wait_until_idle(poll_interval=2.0)
        return True

    training_mod._wait_for_vllm_drain = _blackbox_drain
    logger.info("[Blackbox] Disabled vLLM health checks, using blackbox queue drain")


if __name__ == "__main__":
    pool = _setup_blackbox_pool(PROFILE_PATH)
    _patch_inframind_for_blackbox(pool)

    # Enable response cache (persists across runs)
    # Parse --no-cache before delegating to main()
    use_cache = "--no-cache" not in sys.argv
    if use_cache:
        # Check for --cache arg
        cache_path = DEFAULT_CACHE_PATH
        for i, arg in enumerate(sys.argv):
            if arg == "--cache" and i + 1 < len(sys.argv):
                cache_path = sys.argv[i + 1]
                break
        cache = ResponseCache(cache_path)
        enable_cache(cache)

    # Clean up custom args before passing to training.py's argparser
    clean_argv = []
    skip_next = False
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--no-cache":
            continue
        if arg == "--cache":
            skip_next = True
            continue
        clean_argv.append(arg)
    sys.argv = clean_argv

    # Run the shared training loop
    from MAR.InfraMind.training import main
    main(default_dataset="math")

    if use_cache:
        cache.log_stats()
