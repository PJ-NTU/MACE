"""OpenRouter LLM client (OpenAI-compatible).

Tracks call count + (best-effort) cost; refuses new calls past max_calls.
Retries on transient failures with exponential backoff.
"""
from __future__ import annotations
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Thread-safe OpenRouter client. Concurrent callers may invoke `chat`
    from multiple threads; counters and budget checks are guarded by a lock.
    The HTTP call itself is not in the lock (so calls truly run in parallel).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-3.1-flash-lite",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        max_calls: int = 10_000,
        max_retries: int = 4,
        base_backoff_s: float = 2.0,
        timeout_s: float = 120.0,
    ):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout_s,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_calls = max_calls
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s

        self._lock = threading.Lock()
        self.n_calls = 0
        self.n_failed_calls = 0
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0

    def _reserve_call(self) -> None:
        with self._lock:
            if self.n_calls + self.n_failed_calls >= self.max_calls:
                raise RuntimeError(f"LLM call budget exhausted ({self.max_calls})")

    def _record_success(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.n_calls += 1
            self.prompt_tokens_total += prompt_tokens
            self.completion_tokens_total += completion_tokens

    def _record_failure(self) -> None:
        with self._lock:
            self.n_failed_calls += 1

    def chat(self, prompt: str, system: Optional[str] = None) -> str:
        self._reserve_call()

        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
                ct = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
                choice = resp.choices[0]
                content = choice.message.content
                if content is None or content == "":
                    raise RuntimeError(f"empty completion (finish_reason={choice.finish_reason})")
                self._record_success(pt, ct)
                return content
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                sleep_s = self.base_backoff_s * (2 ** attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s -- sleeping %.1fs",
                    attempt + 1, self.max_retries + 1, e, sleep_s,
                )
                time.sleep(sleep_s)

        self._record_failure()
        raise RuntimeError(f"LLM call failed after {self.max_retries + 1} attempts: {last_err}")

    def stats(self) -> dict:
        with self._lock:
            return {
                "model": self.model,
                "n_calls": self.n_calls,
                "n_failed_calls": self.n_failed_calls,
                "prompt_tokens_total": self.prompt_tokens_total,
                "completion_tokens_total": self.completion_tokens_total,
            }
