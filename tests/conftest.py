import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def scenarios_path():
    return FIXTURES / "scenarios_demo.jsonl"


@pytest.fixture
def knowledge_dir():
    return FIXTURES / "knowledge"


@pytest.fixture
def scenarios(scenarios_path):
    from vskc.dataio import load_scenarios

    return load_scenarios(scenarios_path)


@pytest.fixture
def knowledge(knowledge_dir):
    from vskc.dataio import load_knowledge

    return load_knowledge(knowledge_dir)


@pytest.fixture(autouse=True)
def no_real_key(monkeypatch, tmp_path):
    """Tests never call a model and never read the real key."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("VSKC_KEY_FILE", str(tmp_path / "absent_key.txt"))
    monkeypatch.setenv("VSKC_MOCK", "1")
