"""Provider implementation for OpenRouter."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import langextract as lx
from openai import OpenAI

# Whitelisted OpenRouter params we will forward to the API.
_OPENROUTER_ALLOWED = {
    "stop",                 # str | list[str]
    "max_tokens",           # int
    "temperature",          # float
    "top_p",                # float
    "seed",                 # int
    "top_k",                # int
    "frequency_penalty",    # float
    "presence_penalty",     # float
    "repetition_penalty",   # float
    "min_p",                # float
    "top_a",                # float
    "logit_bias",           # dict
    "top_logprobs",         # int
    "user",                 # str
    # NOTE: We intentionally do NOT proxy `stream` here because this provider
    # returns non-streaming results via LangExtract's generator interface.
}

def _sanitize_openrouter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter kwargs to only those OpenRouter supports. Logs and drops anything else.
    Also quietly disables 'stream=True' if present, since this provider isn't streaming.
    """
    merged = dict(kwargs) if kwargs else {}

    # Drop/disable streaming if user passed it
    if merged.get("stream") is True:
        logging.warning("[openrouter-provider] 'stream=True' requested but this provider is non-streaming; disabling.")
        merged.pop("stream", None)

    clean: Dict[str, Any] = {k: v for k, v in merged.items() if k in _OPENROUTER_ALLOWED}
    dropped = set(merged.keys()) - set(clean.keys())
    if dropped:
        logging.debug(f"[openrouter-provider] Dropping unsupported params: {sorted(dropped)}")
    return clean


def _map_response_format(format_type: Optional[str], json_schema: Optional[dict]) -> Optional[dict]:
    """
    Translate LangExtract's ad-hoc 'format_type' into OpenAI/OpenRouter 'response_format'.
    Supports:
      - format_type="json"        → {"type": "json_object"}
      - format_type="json_schema" → {"type": "json_schema", "json_schema": {...}}
    Returns None if no structured format requested.
    """
    if not format_type:
        return None

    if format_type == "json":
        return {"type": "json_object"}

    if format_type == "json_schema":
        if not isinstance(json_schema, dict):
            logging.warning("[openrouter-provider] format_type='json_schema' requested but no/invalid schema provided; using empty schema.")
            json_schema = {}
        return {"type": "json_schema", "json_schema": json_schema}

    # Unknown format_type → ignore but log
    logging.debug(f"[openrouter-provider] Unknown format_type={format_type!r}; ignoring.")
    return None


def _extract_output(result: Any, prefer_json: bool) -> str:
    """
    Robustly harvest useful text from the OpenAI/OpenRouter response:
      1) choices[0].message.content
      2) choices[0].message.tool_calls[0].function.arguments
      3) choices[0].text (legacy completions)
      4) fallback: '{}' if prefer_json else ''
    """
    try:
        if not result or not getattr(result, "choices", None):
            raise ValueError("No choices in result")

        choice = result.choices[0]
        # 1) chat content
        message = getattr(choice, "message", None)
        if message:
            content = getattr(message, "content", None)
            if content:
                return content

            # 2) tool/function call arguments
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                try:
                    fn = getattr(tool_calls[0], "function", None)
                    if fn:
                        args = getattr(fn, "arguments", None)
                        if args:
                            return args
                except Exception as e:
                    logging.debug(f"[openrouter-provider] tool_calls parsing error ignored: {e}")

        # 3) legacy 'text'
        text = getattr(choice, "text", None)
        if text:
            return text

    except Exception as e:
        logging.debug(f"[openrouter-provider] Response extraction fallback: {e}")

    # 4) final fallback
    return "{}" if prefer_json else ""


def _safe_error_json(code: int, message: str, model_id: str) -> str:
    """Always return valid JSON with a clear provider-prefixed error."""
    safe_msg = f"openrouter-provider: {message}"
    # Use repr to safely quote the message
    return (
        '{"_lx_error":{"provider":"openrouter",'
        f'"model":"{model_id}","code":{code},"message":{repr(safe_msg)}}}'
    )

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_HTTP_REFERER = os.environ.get("OPENROUTER_HTTP_REFERER", "https://github.com/Gutenback/langextract-openrouter")
OPENROUTER_APP_TITLE = os.environ.get("OPENROUTER_APP_TITLE", "LangExtract OpenRouter Provider")

def _mask(key: str | None, keep: int = 6) -> str:
    if not key:
        return "<none>"
    return key[:keep] + "…(" + str(len(key)) + " chars)"

@lx.providers.registry.register(r"^openrouter", priority=10)
class openrouterLanguageModel(lx.inference.BaseLanguageModel):
    """LangExtract provider for OpenRouter-compatible OpenAI API."""

    def __init__(self, model_id: str, api_key: str | None = None, **kwargs: Any):
        super().__init__()
        self.model_id = model_id[11:] if model_id.startswith("openrouter/") else model_id
        self.original_model_id = model_id
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

        self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=OPENROUTER_BASE_URL,
                    default_headers={
                        "Authorization": f"Bearer {self.api_key or ''}",
                        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
                        "X-Title": OPENROUTER_APP_TITLE,
                    },
                )
        self._extra_kwargs = kwargs or {}
        self._authed_ok = False  # lazy auth flag
        logging.info(f"[openrouter-provider] Initialized OpenRouter provider for model: {self.model_id}")

    def _preflight_auth(self):
        """One-time auth sanity check for clearer failures."""
        if self._authed_ok:
            return
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Export OPENROUTER_API_KEY or pass api_key=..."
            )
        self._authed_ok = True

    def _build_call_params(self, **runtime_kwargs: Any) -> tuple[dict, bool]:
        """
        Merge constructor kwargs + runtime kwargs, sanitize for OpenRouter,
        map response_format from format_type/json_schema, and return:
          (call_params, prefer_json_bool)
        """
        merged = {**self._extra_kwargs, **(runtime_kwargs or {})}
        format_type = merged.get("format_type")
        json_schema = merged.get("json_schema")

        # Sanitize forwardable params first
        call_params = _sanitize_openrouter_kwargs(merged)

        # Map response_format, if any
        response_format = _map_response_format(format_type, json_schema)
        if response_format:
            call_params["response_format"] = response_format

        prefer_json = format_type in ("json", "json_schema")
        return call_params, prefer_json

    def infer(self, batch_prompts, **kwargs):
        """
        Run inference on a batch of prompts, guaranteeing non-empty outputs:
          - JSON modes → '{}' if model returns nothing
          - Text modes → default to JSON '{}' for safety (or text if text_error_fallback=True)
        """
        call_params, prefer_json = self._build_call_params(**kwargs)
        text_error_fallback = bool(kwargs.get("text_error_fallback", False))

        # Ensure we have credentials before the first call
        try:
            self._preflight_auth()
        except Exception as e:
            err_json = _safe_error_json(401, str(e), self.model_id)
            for _ in batch_prompts:
                yield [lx.inference.ScoredOutput(score=0.0, output=err_json)]
            return

        for prompt in batch_prompts:
            try:
                logging.info(f"[openrouter-provider] Calling model={self.model_id} with params={call_params!r}")
                result = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": str(prompt)}],
                    **call_params,
                )

                output = _extract_output(result, prefer_json=prefer_json)

                # Final guard against empty strings reaching LangExtract
                if not (output or "").strip():
                    # Prefer a valid JSON object to avoid parser failures upstream.
                    output = "{}" if prefer_json else ("[EMPTY]" if text_error_fallback else "{}")

                yield [lx.inference.ScoredOutput(score=1.0, output=output)]

            except Exception as e:
                logging.error(f"[openrouter-provider] Error calling completion (model={self.model_id}): {e}")
                # Try to extract HTTP status; default to 500
                code = getattr(getattr(e, "response", None), "status_code", None) or 500
                fallback = _safe_error_json(int(code), str(e), self.model_id)
                if text_error_fallback and not prefer_json:
                    fallback = f"openrouter-provider: {e}"
                yield [lx.inference.ScoredOutput(score=0.0, output=fallback)]
