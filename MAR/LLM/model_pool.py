"""
Model pool abstraction for whitebox (vLLM), blackbox (API), and hybrid pools.

Each pool type provides a unified interface for:
  - Model list and metadata
  - Base URL / API key resolution per model
  - Pricing information

The pool does NOT handle metrics — that's the responsibility of the
metrics provider (metrics_watcher for whitebox, blackbox_metrics for blackbox).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class ModelPool:
    """Unified model pool loaded from a JSON profile.

    Works for whitebox (vLLM), blackbox (API), or hybrid configurations.
    The ``pool_type`` field in the JSON determines how models are resolved.
    """

    def __init__(self, profile_path: str) -> None:
        self.profile_path = Path(profile_path)
        with self.profile_path.open("r", encoding="utf-8") as f:
            self._data = json.load(f)

        self.pool_type: str = self._data.get("pool_type", "whitebox")
        self._global = self._data.get("global_settings", {})
        self._models_raw: List[dict] = self._data.get("models", [])

        # Derived fields
        self.models: List[str] = [m["Name"] for m in self._models_raw]
        self.n_models: int = len(self.models)

        # Per-model metadata lookup
        self._model_map: Dict[str, dict] = {m["Name"]: m for m in self._models_raw}

        logger.info(
            "[ModelPool] Loaded {} pool with {} models from {}",
            self.pool_type, self.n_models, self.profile_path.name,
        )

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------

    def get_model_config(self, model_name: str) -> dict:
        return self._model_map.get(model_name, {})

    def get_max_model_len(self, model_name: str) -> int:
        cfg = self._model_map.get(model_name, {})
        return cfg.get("MaxModelLen", self._global.get("default_max_model_len", 16384))

    def get_max_output_tokens(self, model_name: str) -> int:
        cfg = self._model_map.get(model_name, {})
        return cfg.get("MaxOutputTokens", self._global.get("default_max_output_tokens", 2048))

    def get_provider(self, model_name: str) -> str:
        return self._model_map.get(model_name, {}).get("provider", "vllm")

    def is_blackbox(self, model_name: str) -> bool:
        return self.get_provider(model_name) != "vllm"

    # ------------------------------------------------------------------
    # URL / key resolution
    # ------------------------------------------------------------------

    def get_base_url(self, model_name: str) -> Optional[str]:
        cfg = self._model_map.get(model_name, {})

        if self.pool_type == "whitebox":
            # vLLM: use model_base_urls mapping
            urls = self._data.get("model_base_urls", {})
            return urls.get(model_name)

        # Blackbox / hybrid: resolve from env or config
        env_key = cfg.get("base_url_env", "")
        if env_key:
            url = os.environ.get(env_key, "").strip()
            if url:
                return url
        return cfg.get("base_url_default")

    def get_api_key(self, model_name: str) -> str:
        cfg = self._model_map.get(model_name, {})

        if self.pool_type == "whitebox":
            return self._data.get("key", "dummy-key-for-vllm")

        env_key = cfg.get("api_env_key", "")
        if env_key:
            key = os.environ.get(env_key, "").strip()
            if key:
                return key
        return os.environ.get("KEY", "EMPTY")

    def get_rpm_limit(self, model_name: str) -> int:
        return self._model_map.get(model_name, {}).get("rpm_limit", 60)

    # ------------------------------------------------------------------
    # Bulk accessors
    # ------------------------------------------------------------------

    def get_base_urls(self) -> Dict[str, str]:
        """Return {model_name: base_url} for all models."""
        urls = {}
        for name in self.models:
            url = self.get_base_url(name)
            if url:
                urls[name] = url
        return urls

    def get_api_keys(self) -> Dict[str, str]:
        """Return {model_name: api_key} for all models."""
        return {name: self.get_api_key(name) for name in self.models}

    def get_rpm_limits(self) -> Dict[str, int]:
        """Return {model_name: rpm_limit} for all models."""
        return {name: self.get_rpm_limit(name) for name in self.models}


    def get_llm_profile(self) -> List[dict]:
        """Return model profile dicts compatible with MAS router's llm_profile format."""
        return [
            {"Name": m["Name"], "Description": m.get("Description", "")}
            for m in self._models_raw
        ]

    # ------------------------------------------------------------------
    # Metrics URL helpers (whitebox only)
    # ------------------------------------------------------------------

    def get_metrics_urls(self) -> Dict[str, str]:
        """Return {model_name: metrics_url} for whitebox (vLLM) models only."""
        urls = {}
        for name in self.models:
            if self.is_blackbox(name):
                continue
            base_url = self.get_base_url(name)
            if not base_url:
                continue
            base = base_url.rstrip("/")
            if base.endswith("/v1"):
                base = base[:-3].rstrip("/")
            urls[name] = f"{base}/metrics"
        return urls
