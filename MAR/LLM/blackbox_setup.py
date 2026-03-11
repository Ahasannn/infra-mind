"""
Setup helpers for blackbox model pools.

Configures environment variables so that ``gpt_chat.py`` can resolve
per-model base URLs and API keys without modification.

Usage (call once at startup, before any LLM calls):

    from MAR.LLM.model_pool import ModelPool
    from MAR.LLM.blackbox_setup import configure_blackbox_env

    pool = ModelPool("MAR/LLM/llm_profile_blackbox.json")
    configure_blackbox_env(pool)
"""

import hashlib
import json
import os
from typing import Optional

from loguru import logger

from MAR.LLM.model_pool import ModelPool


def configure_blackbox_env(pool: ModelPool) -> None:
    """Set MODEL_BASE_URLS and per-provider API keys from a ModelPool config.

    This patches the environment so ``gpt_chat.py``'s existing resolution
    logic routes each model to the correct provider endpoint.

    For providers that share a base URL but use different API keys (e.g.,
    Gemini vs OpenRouter), we append the API key as a query parameter
    to create distinct (base_url, api_key) pairs for the connection pool.
    """
    base_urls = pool.get_base_urls()
    api_keys = pool.get_api_keys()

    # Set MODEL_BASE_URLS as JSON string for gpt_chat.py
    os.environ["MODEL_BASE_URLS"] = json.dumps(base_urls)
    logger.info("[BlackboxSetup] MODEL_BASE_URLS set for {} models", len(base_urls))

    # For blackbox, we need per-provider API keys.
    # gpt_chat.py uses a single KEY env var. We'll use the first provider's key
    # as the default, but also register per-model keys via a custom mechanism.
    #
    # Strategy: since _get_shared_sync_client() keys on (base_url, api_key),
    # we store per-provider keys and monkey-patch _resolve_api_key to be
    # model-aware. But to keep it simple, we store a mapping and patch once.

    _setup_per_model_api_keys(pool)

    # Register pricing for blackbox models
    _register_blackbox_pricing(pool)

    # Register max context lengths
    _register_blackbox_context_limits(pool)

    for name in pool.models:
        logger.debug(
            "[BlackboxSetup] {} -> {} (provider={})",
            name, base_urls.get(name, "?"), pool.get_provider(name),
        )


def _setup_per_model_api_keys(pool: ModelPool) -> None:
    """Patch _resolve_api_key everywhere to support per-model keys.

    gpt_chat._resolve_api_key() is imported by name into multiple modules
    (gpt_chat, agent, inframind_agent). We use a thread-local to pass the
    current model name so the patched function can return the correct key.
    """
    import threading
    import MAR.LLM.gpt_chat as gpt_chat

    model_keys = pool.get_api_keys()
    _original_resolve = gpt_chat._resolve_api_key

    # Thread-local carries the model name for the current LLM call
    _tls = threading.local()

    def _resolve_api_key_patched() -> str:
        model_name = getattr(_tls, "current_model", None)
        if model_name and model_name in model_keys:
            return model_keys[model_name]
        return _original_resolve()

    # Patch all modules that import _resolve_api_key by name
    gpt_chat._resolve_api_key = _resolve_api_key_patched
    try:
        import MAR.Agent.agent as agent_mod
        agent_mod._resolve_api_key = _resolve_api_key_patched
    except ImportError:
        pass
    try:
        import MAR.InfraMind.inframind_agent as im_agent_mod
        im_agent_mod._resolve_api_key = _resolve_api_key_patched
    except ImportError:
        pass
    try:
        import MAR.InfraMind.env as env_mod
        if hasattr(env_mod, "_resolve_api_key"):
            env_mod._resolve_api_key = _resolve_api_key_patched
    except ImportError:
        pass

    # Set thread-local before every LLM call so the resolver knows which model
    def _wrap_gen(original):
        def wrapped(self, *args, **kwargs):
            _tls.current_model = self.model_name
            try:
                return original(self, *args, **kwargs)
            finally:
                _tls.current_model = None
        return wrapped

    def _wrap_agen(original):
        async def wrapped(self, *args, **kwargs):
            _tls.current_model = self.model_name
            try:
                return await original(self, *args, **kwargs)
            finally:
                _tls.current_model = None
        return wrapped

    gpt_chat.ALLChat.gen = _wrap_gen(gpt_chat.ALLChat.gen)
    gpt_chat.ALLChat.agen = _wrap_agen(gpt_chat.ALLChat.agen)

    # ── Patch _execute to set cache metadata on agents ─────────────
    # Before each _call_llm_stream, the agent needs to know:
    #   - query_hash: hash of raw query text (item identifier)
    #   - has_upstream: whether spatial/temporal inputs exist
    #   - call_seq: 0 for first call, incremented for Reflection 2nd call
    #   - role/topology/strategy: from agent attributes
    # This metadata is set in _execute and read by _call_llm_stream wrapper.

    def _make_execute_wrapper(original_execute):
        """Wrap _execute to set _cache_meta before LLM calls."""
        def wrapper(self, input, spatial_info, temporal_info, **kwargs):
            query = input.get("query", "")
            query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
            has_upstream = bool(spatial_info) or bool(temporal_info)
            role_name = ""
            if hasattr(self, "role") and hasattr(self.role, "role"):
                role_name = self.role.role
            self._cache_meta = {
                "query_hash": query_hash,
                "has_upstream": has_upstream,
                "call_seq": 0,
                "role_name": role_name,
                "topology": getattr(self, "reason", ""),
                "strategy": getattr(self, "strategy_name", ""),
            }
            return original_execute(self, input, spatial_info, temporal_info, **kwargs)
        return wrapper

    def _make_finalrefer_execute_wrapper(original_execute):
        """Wrap FinalRefer._execute — always has upstream (spatial from agents)."""
        def wrapper(self, input, spatial_info, temporal_info, **kwargs):
            self._cache_meta = {
                "has_upstream": True,  # FinalRefer always aggregates agent outputs
                "call_seq": 0,
            }
            return original_execute(self, input, spatial_info, temporal_info, **kwargs)
        return wrapper

    # ── Patch _call_llm_stream with cache + API key + queue ──────
    def _make_queued_wrapper(original_fn):
        """Wrap _call_llm_stream with deterministic cache + API key + RPM queue.

        Cache key = (model, role, topology, strategy, query_hash).
        Only caches when has_upstream=False and call_seq==0.
        On hit: go through RPM gate + sleep for stored latency.
        On miss: go through RPM gate + real API call + store in cache.
        """
        def wrapper(self, *args, **kwargs):
            import time as _time
            from MAR.InfraMind.blackbox_queue import BlackboxRequestQueue
            from MAR.LLM.response_cache import get_cache, make_cache_key

            _tls.current_model = self.llm_name
            meta = getattr(self, "_cache_meta", None)
            cache = get_cache()

            # Determine if this call is cacheable:
            #   - cache enabled, meta set by _execute wrapper
            #   - no upstream context (no spatial/temporal inputs)
            #   - first call in _execute (call_seq==0, not Reflection 2nd call)
            cacheable = (
                cache is not None
                and meta is not None
                and not meta.get("has_upstream", True)
                and meta.get("call_seq", 1) == 0
            )

            # Always increment call_seq so Reflection 2nd call sees call_seq=1
            if meta is not None:
                seq = meta.get("call_seq", 0)
                meta["call_seq"] = seq + 1

            # ── Cache hit path ───────────────────────────────────────
            cache_key = None
            if cacheable:
                cache_key = make_cache_key(
                    model_name=self.llm_name,
                    role_name=meta.get("role_name", ""),
                    topology_name=meta.get("topology", ""),
                    strategy_name=meta.get("strategy", ""),
                    query_hash=meta.get("query_hash", ""),
                )
                cached = cache.get(cache_key, model_name=self.llm_name)
                if cached is not None:
                    response_text, latency = cached
                    # Simulate API timing: RPM gate + sleep
                    queue = BlackboxRequestQueue.instance()
                    try:
                        if queue is not None:
                            deadline = float(getattr(self, "priority", None) or float("inf"))
                            def _sleep_fn():
                                _time.sleep(latency)
                                return response_text
                            return queue.submit(self.llm_name, _sleep_fn, deadline=deadline)
                        else:
                            _time.sleep(latency)
                            return response_text
                    finally:
                        _tls.current_model = None

            # ── Cache miss / not cacheable — real API call ───────────
            try:
                queue = BlackboxRequestQueue.instance()
                # Measure only the API call time (excluding queue wait)
                # so cached latency reflects pure model response time.
                api_latency_box = [0.0]
                if queue is not None:
                    deadline = float(getattr(self, "priority", None) or float("inf"))
                    def _timed_call():
                        t0 = _time.perf_counter()
                        r = original_fn(self, *args, **kwargs)
                        api_latency_box[0] = _time.perf_counter() - t0
                        return r
                    result = queue.submit(
                        self.llm_name,
                        _timed_call,
                        deadline=deadline,
                    )
                else:
                    t0 = _time.perf_counter()
                    result = original_fn(self, *args, **kwargs)
                    api_latency_box[0] = _time.perf_counter() - t0
                api_latency = api_latency_box[0]

                # Store in cache (only for cacheable misses)
                # Latency is pure API time — queue wait is NOT included.
                if cacheable and cache_key is not None and result:
                    cache.put(
                        cache_key, self.llm_name,
                        meta.get("role_name", ""),
                        meta.get("topology", ""),
                        meta.get("strategy", ""),
                        result, api_latency,
                    )
                return result
            finally:
                _tls.current_model = None
        return wrapper

    # Apply patches to Agent
    try:
        from MAR.Agent.agent import Agent, FinalRefer
        Agent._execute = _make_execute_wrapper(Agent._execute)
        Agent._call_llm_stream = _make_queued_wrapper(Agent._call_llm_stream)
        FinalRefer._execute = _make_finalrefer_execute_wrapper(FinalRefer._execute)
        FinalRefer._call_llm_stream = _make_queued_wrapper(FinalRefer._call_llm_stream)
    except ImportError:
        pass

    # Apply patches to InfraMindAgent
    try:
        from MAR.InfraMind.inframind_agent import InfraMindAgent
        InfraMindAgent._execute = _make_execute_wrapper(InfraMindAgent._execute)
        InfraMindAgent._call_llm_stream = _make_queued_wrapper(InfraMindAgent._call_llm_stream)
    except ImportError:
        pass

    logger.info("[BlackboxSetup] Per-model API key resolution enabled for {} models", len(model_keys))


def _register_blackbox_pricing(pool: ModelPool) -> None:
    """Add blackbox model pricing to the global MODEL_PRICE dict."""
    from MAR.LLM.price import MODEL_PRICE

    for name in pool.models:
        if name not in MODEL_PRICE:
            cfg = pool.get_model_config(name)
            pricing = cfg.get("pricing", {"input": 0.50, "output": 1.00})
            MODEL_PRICE[name] = pricing
            logger.debug("[BlackboxSetup] Registered pricing for {}: {}", name, pricing)


def _register_blackbox_context_limits(pool: ModelPool) -> None:
    """Register max context/output limits for blackbox models."""
    from MAR.LLM import llm_profile_full

    for name in pool.models:
        if name not in llm_profile_full._MAX_MODEL_LEN_MAP:
            llm_profile_full._MAX_MODEL_LEN_MAP[name] = pool.get_max_model_len(name)
        if name not in llm_profile_full._MAX_OUTPUT_TOKENS_MAP:
            llm_profile_full._MAX_OUTPUT_TOKENS_MAP[name] = pool.get_max_output_tokens(name)
