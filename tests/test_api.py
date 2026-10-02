"""The web API, against fixture data and a mock run. No model is called."""
import json

import pytest
from fastapi.testclient import TestClient

import run_eval
from vskc.api import create_app
from vskc.dataio import load_scenarios, write_jsonl


@pytest.fixture
def client(tmp_path, scenarios_path, knowledge_dir):
    scen_dir = tmp_path / "scenarios"
    scen_dir.mkdir()
    rows = []
    for s in load_scenarios(scenarios_path):
        d = s.model_dump(mode="json")
        d["meta"].update(label_certainty="clear", depends_on_supplier_fact=s.id == "demo-003")
        rows.append(d)
    write_jsonl(scen_dir / "heldout.jsonl", rows)
    results = tmp_path / "results"
    assert run_eval.main(["--split", "custom", "--scenarios", str(scen_dir / "heldout.jsonl"),
                          "--knowledge-dir", str(knowledge_dir), "--results-dir", str(results),
                          "--mock", "--run-id", "mock-api"]) == 0
    # present the mock run as a real held-out run so the read paths are exercised
    meta_p = results / "runs" / "mock-api" / "run_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta.update(mock=False, split="heldout",
                gate={"tau_conf": 0.5, "tau_ret": 0.0, "selected_on_run": "x", "rule": "test"})
    meta_p.write_text(json.dumps(meta), encoding="utf-8")
    (results / "gate.json").write_text(json.dumps(meta["gate"]), encoding="utf-8")
    app = create_app(results_dir=results, scenarios_dir=scen_dir, gold_dir=tmp_path / "gold",
                     knowledge_dir=knowledge_dir)
    return TestClient(app)


def test_index_and_static(client):
    assert "Vertical Sales Knowledge Compiler" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_meta_has_no_secret(client):
    r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    assert len(body["actions"]) == 9 and body["knowledge_count"] == 5
    assert "sk-or-" not in r.text and "api_key" not in r.text


def test_scenario_list_and_detail(client):
    lst = client.get("/api/scenarios?split=heldout").json()
    assert len(lst["items"]) == 6 and lst["run_id"] == "mock-api"
    assert set(lst["items"][0]["recorded"]) == {"rule", "llm", "rag"}
    d = client.get("/api/scenarios/heldout/demo-003").json()
    assert d["label"] == "disqualify" and d["depends_on_supplier_fact"] is True
    outs = d["recorded_run"]["outputs"]
    assert [o["system"] for o in outs] == ["rule", "llm", "rag"]
    rag = outs[2]
    assert rag["retrieved"] and all("text" in r for r in rag["retrieved"])
    assert rag["gate"]["decision"] in ("pass", "human_review")
    assert outs[0]["gate"] is None and outs[0]["cost_usd"] == 0.0
    assert client.get("/api/scenarios/heldout/nope").status_code == 404


def test_recommend_mock(client):
    body = {"mode": "mock", "scenario": {"narrative": "A buyer needs ATEX certified proximity sensors.",
                                         "sales_stage": "qualifying", "customer_size": "small",
                                         "objections": ["ATEX certification"]}}
    r = client.post("/api/recommend", json=body)
    assert r.status_code == 200
    outs = r.json()["outputs"]
    assert [o["system"] for o in outs] == ["rule", "llm", "rag"]
    assert outs[0]["action"] == "disqualify"
    assert all(o["correct"] is None for o in outs)


def test_recommend_rejects_bad_input(client):
    assert client.post("/api/recommend", json={"mode": "mock", "scenario": {"narrative": "",
                       "sales_stage": "qualifying", "customer_size": "small"}}).status_code == 422
    assert client.post("/api/recommend", json={"mode": "mock", "scenario": {"narrative": "x",
                       "sales_stage": "lunch", "customer_size": "small"}}).status_code == 422


def test_results_summary(client):
    res = client.get("/api/results").json()
    h = res["heldout_raw"]
    assert h["run_id"] == "mock-api" and h["complete"] is True
    assert set(h["systems"]) == {"rule", "llm", "rag"}
    assert "all" in h["slices"] and "depends_on_supplier_fact" in h["slices"]
    assert any(p["a"] == "rag" and p["b"] == "llm" and p["slice"] == "all" for p in h["paired"])
    assert 0 <= h["gate"]["abstain_rate"] <= 1


def test_knowledge_endpoint(client):
    assert client.get("/api/knowledge/fx-pb-003").json()["type"] == "playbook"
    assert client.get("/api/knowledge/none").status_code == 404
