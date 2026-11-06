"""Minimal LM Studio client using only the standard library (no openai package).

=============================================================================
READ ME FIRST (remote desktop / Windows)
=============================================================================
What this file is:
  The connection to LM Studio's local server. You never run this file
  directly; run_generation.py creates an LMStudioClient from it.

What it depends on:
  - LM Studio, with a model LOADED and the local server STARTED:
      LM Studio -> "Developer" tab (the >_ icon) -> load a model ->
      toggle "Status: Running". The server address shown there
      (default http://localhost:1234) must match LLM_BASE_URL in
      run_generation.py, with "/v1" on the end.
  - Python standard library only, nothing to pip install.

Troubleshooting:
  "cannot reach LM Studio ..."  -> the server is not started, or the port
      in LLM_BASE_URL differs from the one LM Studio shows.
  "timed out"                   -> the model is too slow for one request;
      raise LLM_TIMEOUT_SECONDS in run_generation.py or use a smaller model.
  HTTP 400 "context length"     -> raise the model's Context Length in
      LM Studio's load settings (8192+ is recommended for long chunks).

Mimics the small slice of openai.OpenAI that generate_sample.py uses:
    client.chat.completions.create(model=..., messages=[...]) -> .choices[0].message.content
    client.models.list() -> .data[i].id
=============================================================================
"""

import json
import socket
import urllib.error
import urllib.request

# Talk to the server directly: Windows machines often have a system proxy
# configured, and routing localhost requests through it fails.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class _Obj:
    """Wrap a JSON dict so fields read as attributes (response.choices[0].message.content)."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = _Obj(value)
            elif isinstance(value, list):
                value = [_Obj(v) if isinstance(v, dict) else v for v in value]
            setattr(self, key, value)


class LMStudioError(Exception):
    """Raised when the LM Studio server returns an error for one request."""


class LMStudioConnectionError(LMStudioError, ConnectionError):
    """Raised when the LM Studio server cannot be reached at all. Subclasses
    ConnectionError so generate_sample.py stops the run instead of retrying."""


class LMStudioClient:
    def __init__(self, base_url: str = "http://localhost:1234/v1", timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.chat = _Chat(self)
        self.models = _Models(self)

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _OPENER.open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise LMStudioError(f"HTTP {e.code} from {path}: {detail}") from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise LMStudioError(f"request to {path} timed out after {self.timeout}s") from e
            raise LMStudioConnectionError(
                f"cannot reach LM Studio at {self.base_url} ({e.reason}); "
                "is the local server started?"
            ) from e
        except (socket.timeout, TimeoutError) as e:
            raise LMStudioError(f"request to {path} timed out after {self.timeout}s") from e


class _Completions:
    def __init__(self, client: LMStudioClient):
        self._client = client

    def create(self, model: str, messages: list[dict], **kwargs) -> _Obj:
        body = {"model": model, "messages": messages, **kwargs}
        return _Obj(self._client._request("POST", "/chat/completions", body))


class _Chat:
    def __init__(self, client: LMStudioClient):
        self.completions = _Completions(client)


class _Models:
    def __init__(self, client: LMStudioClient):
        self._client = client

    def list(self) -> _Obj:
        return _Obj(self._client._request("GET", "/models"))
