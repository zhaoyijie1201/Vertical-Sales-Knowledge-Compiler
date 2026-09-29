"""The single entry point for model calls.

Every attempt, successful or not, is appended to results/raw/<run_id>.jsonl so that all
reported numbers can be recomputed without an API key. The key itself is never logged.
"""
import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import Settings, api_key
from .dataio import append_jsonl

_client = None


def _get_client(settings: Settings):
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(base_url=settings.base_url, api_key=api_key())
    return _client


def extract_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object from a model reply, tolerating code fences and surrounding prose."""
    if text is None:
        raise ValueError("empty reply")
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in reply")
        obj = json.loads(t[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    return obj


def _mock_value(spec: Dict[str, Any], h: int, user: str, name: str) -> Any:
    t = spec.get("type")
    if "enum" in spec:
        return spec["enum"][h % len(spec["enum"])]
    if t == "string":
        return "mock %s" % name
    if t == "number":
        return round(0.30 + ((h >> 8) % 70) / 100.0, 2)
    if t == "integer":
        return h % 10
    if t == "boolean":
        return bool(h % 2)
    if t == "array":
        if name == "evidence_ids":
            return re.findall(r'<item id="([^"]+)"', user)[:2]
        return []
    if t == "object":
        return {k: _mock_value(v, h, user, k) for k, v in spec.get("properties", {}).items()}
    return None


def _mock_completion(user: str, schema: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
    """Deterministic fake reply. It exercises the pipeline and carries no information."""
    h = int(hashlib.sha256(user.encode("utf-8")).hexdigest(), 16)
    obj = _mock_value(schema, h, user, "root")
    text = json.dumps(obj, ensure_ascii=False)
    return text, {"prompt_tokens": len(user) // 4, "completion_tokens": len(text) // 4}


def _response_format(settings: Settings, schema: Dict[str, Any], schema_name: str) -> Optional[Dict[str, Any]]:
    if settings.response_format == "json_schema":
        return {"type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
    if settings.response_format == "json_object":
        return {"type": "json_object"}
    return None


def complete_json(
    *,
    settings: Settings,
    model: str,
    system: str,
    user: str,
    schema: Dict[str, Any],
    schema_name: str,
    validate: Callable[[Dict[str, Any]], Any],
    run_id: str,
    raw_dir: Path,
    tag: Dict[str, Any],
    max_attempts: int = 2,
) -> Tuple[Optional[Any], Dict[str, Any], Optional[str]]:
    """Call the model and return (validated object or None, stats, error or None).

    A reply that fails parsing or validation is retried once with the error appended.
    """
    if not settings.mock and not model:
        raise RuntimeError("No model configured. Set the model slug in .env or run with --mock.")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    stats = {"input_tokens": 0, "output_tokens": 0, "latency_s": 0.0, "attempts": 0}
    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        raw: Optional[str] = None
        usage: Dict[str, Any] = {}
        error: Optional[str] = None
        result = None
        try:
            if settings.mock:
                raw, usage = _mock_completion(user, schema)
            else:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": settings.max_tokens,
                }
                rf = _response_format(settings, schema, schema_name)
                if rf is not None:
                    kwargs["response_format"] = rf
                resp = _get_client(settings).chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content
                if resp.usage is not None:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                    }
            result = validate(extract_json(raw))
        except Exception as e:  # logged and surfaced to the caller; never swallowed silently
            error = "%s: %s" % (type(e).__name__, str(e)[:500])

        latency = round(time.time() - t0, 3)
        stats["attempts"] = attempt
        stats["latency_s"] = round(stats["latency_s"] + latency, 3)
        stats["input_tokens"] += int(usage.get("prompt_tokens") or 0)
        stats["output_tokens"] += int(usage.get("completion_tokens") or 0)

        record = {
            "call_id": str(uuid.uuid4()),
            "run_id": run_id,
            "attempt": attempt,
            "model": "mock" if settings.mock else model,
            "mock": settings.mock,
            "response_format": settings.response_format,
            "system": system,
            "messages_after_system": messages[1:],
            "raw": raw,
            "usage": usage,
            "latency_s": latency,
            "error": error,
        }
        record.update(tag)
        append_jsonl(Path(raw_dir) / ("%s.jsonl" % run_id), record)

        if error is None:
            return result, stats, None

        last_error = error
        if raw is not None:
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user",
                 "content": "That reply was not accepted (%s). Return only the JSON object." % error},
            ]

    return None, stats, last_error
