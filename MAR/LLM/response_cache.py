"""
Model-level LLM response cache for blackbox experiments.

Caches individual LLM call responses keyed on EXPLICIT deterministic fields:
  (model_name, role_name, topology_name, strategy_name, query_hash)

Only caches calls with NO upstream context (no spatial/temporal agent inputs):
  - IO / CoT: single call, always cacheable
  - Reflection: call 1 only (call 2 depends on call 1 output)
  - Debate round 0: each agent's independent call
  - Chain agent 1: first in chain, no upstream
  - FinalRefer: NOT cached (always has upstream from other agents)

On cache hit:
  - Sleep for the stored real latency (simulates API call timing)
  - Still goes through RPM gate (realistic queue pressure)
  - Returns cached response text

On cache miss:
  - Makes real API call
  - Stores (response, latency) in SQLite
  - Returns response

Storage: SQLite (WAL mode) for concurrent safety.
"""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

# ── Global cache singleton ──────────────────────────────────────────
_global_cache: Optional["ResponseCache"] = None


def enable_cache(cache: "ResponseCache") -> None:
    """Set the global model-level cache (call once at startup)."""
    global _global_cache
    _global_cache = cache
    logger.info("[Cache] Global model-level cache enabled")


def get_cache() -> Optional["ResponseCache"]:
    """Return the global cache instance, or None if caching is disabled."""
    return _global_cache


def make_cache_key(
    model_name: str,
    role_name: str,
    topology_name: str,
    strategy_name: str,
    query_hash: str,
) -> str:
    """Deterministic cache key from explicit named fields.

    These fields fully identify a cacheable LLM call (one with no upstream
    agent context).  The same (item, role, model, strategy, topology)
    always produces the same prompt → same response.
    """
    raw = json.dumps(
        {
            "m": model_name,
            "r": role_name,
            "t": topology_name,
            "s": strategy_name,
            "q": query_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResponseCache:
    """SQLite-backed model-level response cache.

    Caches individual LLM responses so identical prompts across different
    epochs / arrival rates / budget tiers don't require redundant API calls.

    Thread-safe. Supports concurrent readers (WAL mode).
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        self._hits = 0
        self._misses = 0
        # Per-model counters
        self._model_hits: Dict[str, int] = {}
        self._model_misses: Dict[str, int] = {}
        self._lock = threading.Lock()
        logger.info("[Cache] Opened at {} ({} existing entries)", self.db_path, self.size())

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                cache_key       TEXT PRIMARY KEY,
                model_name      TEXT NOT NULL,
                role_name       TEXT NOT NULL,
                topology        TEXT NOT NULL,
                strategy        TEXT NOT NULL,
                response_text   TEXT NOT NULL,
                latency         REAL NOT NULL DEFAULT 0.0,
                created_at      REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON responses (model_name)")
        conn.commit()

    def get(self, cache_key: str, model_name: str = "") -> Optional[Tuple[str, float]]:
        """Look up a cached response. Returns (response_text, latency) or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT response_text, latency FROM responses WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        with self._lock:
            if row is not None:
                self._hits += 1
                if model_name:
                    self._model_hits[model_name] = self._model_hits.get(model_name, 0) + 1
                return row[0], row[1]
            self._misses += 1
            if model_name:
                self._model_misses[model_name] = self._model_misses.get(model_name, 0) + 1
            return None

    def put(
        self,
        cache_key: str,
        model_name: str,
        role_name: str,
        topology: str,
        strategy: str,
        response_text: str,
        latency: float,
    ) -> None:
        """Store an LLM response in the cache."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR IGNORE INTO responses
               (cache_key, model_name, role_name, topology, strategy,
                response_text, latency, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (cache_key, model_name, role_name, topology, strategy,
             response_text, latency, time.time()),
        )
        conn.commit()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def size(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return row[0] if row else 0

    def get_model_stats(self) -> Dict[str, Dict[str, int]]:
        """Return {model: {hits, misses}} for all models seen."""
        with self._lock:
            models = set(self._model_hits.keys()) | set(self._model_misses.keys())
            return {
                m: {
                    "hits": self._model_hits.get(m, 0),
                    "misses": self._model_misses.get(m, 0),
                }
                for m in sorted(models)
            }

    def log_stats(self) -> None:
        logger.info(
            "[Cache] {} entries | {} hits / {} misses ({:.1%} hit rate)",
            self.size(), self._hits, self._misses, self.hit_rate,
        )
        for model, stats in self.get_model_stats().items():
            h, m = stats["hits"], stats["misses"]
            total = h + m
            rate = h / total if total > 0 else 0.0
            logger.info(
                "[Cache]   {} — {} hits / {} misses ({:.1%})",
                model, h, m, rate,
            )

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None
