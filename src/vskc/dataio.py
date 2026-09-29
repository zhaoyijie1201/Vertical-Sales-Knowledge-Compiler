"""JSONL reading and writing, plus typed loaders."""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schema import KnowledgeItem, Scenario


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError("%s line %d is not valid JSON: %s" % (path, lineno, e))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_scenarios(path: Path) -> List[Scenario]:
    out = []
    for i, row in enumerate(read_jsonl(path), 1):
        try:
            out.append(Scenario.model_validate(row))
        except Exception as e:
            raise ValueError("%s record %d (%s): %s" % (path, i, row.get("id"), e))
    ids = [s.id for s in out]
    dup = sorted({x for x in ids if ids.count(x) > 1})
    if dup:
        raise ValueError("%s has duplicate ids: %s" % (path, dup))
    return out


def load_knowledge(directory: Path) -> List[KnowledgeItem]:
    """Load every *.jsonl file in the directory as knowledge items, in a stable order."""
    items: List[KnowledgeItem] = []
    if not directory.exists():
        return items
    for path in sorted(directory.glob("*.jsonl")):
        for i, row in enumerate(read_jsonl(path), 1):
            try:
                items.append(KnowledgeItem.model_validate(row))
            except Exception as e:
                raise ValueError("%s record %d (%s): %s" % (path, i, row.get("id"), e))
    ids = [k.id for k in items]
    dup = sorted({x for x in ids if ids.count(x) > 1})
    if dup:
        raise ValueError("knowledge base has duplicate ids: %s" % dup)
    return items


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(directory: Path, pattern: str = "*.jsonl") -> Dict[str, str]:
    if not directory.exists():
        return {}
    return {p.name: sha256_file(p) for p in sorted(directory.glob(pattern))}
