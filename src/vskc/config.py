"""Settings read from the environment (.env at the repository root).

The API key is never stored in a Settings object and never written to any log.
"""
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import ROOT

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at import time
    load_dotenv = None


def _float(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else None


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    base_url: str
    model_under_test: str
    model_gen_dev: str
    model_gen_heldout: str
    top_k: int
    retrieval: str                # single | fused
    max_tokens: int
    response_format: str          # json_schema | json_object | none
    price_in_per_mtok: Optional[float]
    price_out_per_mtok: Optional[float]
    price_checked_on: str
    mock: bool

    def public_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_settings(mock: bool = False) -> Settings:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
    mock = mock or os.environ.get("VSKC_MOCK", "").strip() in ("1", "true", "yes")
    return Settings(
        base_url=os.environ.get("VSKC_BASE_URL", "https://openrouter.ai/api/v1").strip(),
        model_under_test=os.environ.get("VSKC_MODEL_UNDER_TEST", "").strip(),
        model_gen_dev=os.environ.get("VSKC_MODEL_GEN_DEV", "").strip(),
        model_gen_heldout=os.environ.get("VSKC_MODEL_GEN_HELDOUT", "").strip(),
        top_k=_int("VSKC_TOP_K", 5),
        retrieval=os.environ.get("VSKC_RETRIEVAL", "fused").strip() or "fused",
        max_tokens=_int("VSKC_MAX_TOKENS", 4000),
        response_format=os.environ.get("VSKC_RESPONSE_FORMAT", "json_schema").strip() or "json_schema",
        price_in_per_mtok=_float("VSKC_PRICE_IN_PER_MTOK"),
        price_out_per_mtok=_float("VSKC_PRICE_OUT_PER_MTOK"),
        price_checked_on=os.environ.get("VSKC_PRICE_CHECKED_ON", "").strip(),
        mock=mock,
    )


DEFAULT_KEY_FILE = ROOT.parent / "openrouter_api.txt"
_KEY_PATTERN = re.compile(r"sk-or-[A-Za-z0-9_-]+")


def key_file() -> Path:
    return Path(os.environ.get("VSKC_KEY_FILE", "").strip() or DEFAULT_KEY_FILE)


def api_key() -> str:
    """Resolve the API key: the OPENROUTER_API_KEY variable first, then the key file.

    The key file lives outside this repository so it cannot be committed with it.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        path = key_file()
        if path.is_file():
            text = path.read_text(encoding="utf-8-sig").strip()
            m = _KEY_PATTERN.search(text)
            key = m.group(0) if m else text
    if not key:
        raise RuntimeError(
            "No API key found. Set OPENROUTER_API_KEY in .env, or put the key in %s, "
            "or run with --mock to exercise the pipeline without calling a model." % key_file()
        )
    return key
