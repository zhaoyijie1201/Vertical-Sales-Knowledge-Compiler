"""Small helpers shared by the command line scripts."""
import datetime
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd


def setup_console() -> None:
    """Windows consoles default to a legacy code page; keep output readable."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def pct(x: Optional[float], digits: int = 1) -> str:
    return "n/a" if x is None else "%.*f%%" % (digits, 100.0 * x)


def num(x: Optional[float], digits: int = 2) -> str:
    return "n/a" if x is None else "%.*f" % (digits, x)


def md_table(df: pd.DataFrame) -> str:
    """Markdown table with every value rendered as given (no float reformatting)."""
    return df.astype(str).to_markdown(index=False)


def rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def fail(message: str, code: int = 2) -> int:
    print("ERROR: " + message, file=sys.stderr)
    return code


def section(title: str, body: Sequence[Any]) -> str:
    return "\n".join(["## " + title, ""] + [str(b) for b in body] + [""])
