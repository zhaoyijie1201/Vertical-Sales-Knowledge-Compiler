"""Project paths. Everything is resolved relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
GOLD = DATA / "gold"
KNOWLEDGE = DATA / "knowledge"
SCENARIOS = DATA / "scenarios"

RESULTS = ROOT / "results"


def raw_dir(results_dir: Path) -> Path:
    return results_dir / "raw"


def runs_dir(results_dir: Path) -> Path:
    return results_dir / "runs"


def tables_dir(results_dir: Path) -> Path:
    return results_dir / "tables"
