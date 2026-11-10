"""Minimal LLM clients for the judge (standard library only, no SDK version issues).

Two wire formats cover most providers:
  anthropic          -> POST {base_url}/v1/messages           (x-api-key header)
  openai_compatible  -> POST {base_url}/chat/completions      (Bearer token); works with any provider that
                        offers an OpenAI-compatible endpoint (set base_url, e.g. https://<provider>/v1)

Retries: HTTP 408/409/429/5xx and network errors are retried with exponential backoff (honouring
Retry-After). Other HTTP errors (400/401/403/404) fail immediately with the provider's message.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from configs.config import JudgeSettings


class JudgeError(RuntimeError):
    pass


class _Retryable(Exception):
    def __init__(self, message: str, retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after


def get_api_key(settings: JudgeSettings) -> str:
    key = os.getenv(settings.api_key_env)
    if not key:
        try:                                              # Kaggle Secrets
            from kaggle_secrets import UserSecretsClient # type: ignore[import]
            key = UserSecretsClient().get_secret(settings.api_key_env)
        except Exception:
            key = None
    if not key:
        raise JudgeError(f"No API key found. Set the environment variable or Kaggle secret "
                         f"'{settings.api_key_env}'.")
    return key


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        delay = self._last + self.min_interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


def _post(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        if e.code in (408, 409, 429) or e.code >= 500:
            try:
                retry_after = float(e.headers.get("Retry-After", 0) or 0)
            except ValueError:
                retry_after = 0.0
            raise _Retryable(f"HTTP {e.code}: {body}", retry_after)
        raise JudgeError(f"HTTP {e.code}: {body}")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise _Retryable(f"network error: {e!r}")


class JudgeClient:
    def __init__(self, settings: JudgeSettings, api_key: Optional[str] = None):
        if not settings.model:
            raise JudgeError("eval.llm_judge.model is empty. Set the exact model name from your provider.")
        self.s = settings
        self.key = api_key or get_api_key(settings)
        self.limiter = RateLimiter(settings.requests_per_minute)
        self.calls = 0

    # subclasses build (url, headers, payload) and extract text
    def _request(self, system: str, user: str):
        raise NotImplementedError

    def _extract(self, response: dict) -> str:
        raise NotImplementedError

    def complete(self, system: str, user: str) -> str:
        url, headers, payload = self._request(system, user)
        last: Optional[Exception] = None
        for attempt in range(self.s.max_retries + 1):
            self.limiter.wait()
            try:
                self.calls += 1
                return self._extract(_post(url, headers, payload, self.s.timeout_s))
            except _Retryable as e:
                last = e
                time.sleep(max(e.retry_after, min(2.0 ** attempt, 60.0)))
        raise JudgeError(f"Gave up after {self.s.max_retries + 1} attempts: {last}")


class AnthropicClient(JudgeClient):
    def _request(self, system: str, user: str):
        base = (self.s.base_url or "https://api.anthropic.com").rstrip("/")
        payload = {"model": self.s.model, "max_tokens": self.s.max_tokens, "system": system,
                   "messages": [{"role": "user", "content": user}]}
        if self.s.temperature is not None:                # set temperature: null if your model rejects it
            payload["temperature"] = self.s.temperature
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        return f"{base}/v1/messages", headers, payload

    def _extract(self, response: dict) -> str:
        try:
            return "".join(b.get("text", "") for b in response["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as e:
            raise JudgeError(f"Unexpected response shape: {str(response)[:300]}") from e


class OpenAICompatibleClient(JudgeClient):
    def _request(self, system: str, user: str):
        if not self.s.base_url:
            raise JudgeError("eval.llm_judge.base_url is required for provider: openai_compatible")
        payload = {"model": self.s.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   self.s.max_tokens_param: self.s.max_tokens}
        if self.s.temperature is not None:
            payload["temperature"] = self.s.temperature
        headers = {"Authorization": f"Bearer {self.key}", "content-type": "application/json"}
        return f"{self.s.base_url.rstrip('/')}/chat/completions", headers, payload

    def _extract(self, response: dict) -> str:
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise JudgeError(f"Unexpected response shape: {str(response)[:300]}") from e


def make_client(settings: JudgeSettings, api_key: Optional[str] = None) -> JudgeClient:
    if settings.provider == "anthropic":
        return AnthropicClient(settings, api_key)
    if settings.provider == "openai_compatible":
        return OpenAICompatibleClient(settings, api_key)
    raise JudgeError(f"Unknown provider '{settings.provider}' (use 'anthropic' or 'openai_compatible')")