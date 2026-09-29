"""End-to-end run of the scripts in mock mode. No model is called."""
import json

import leakage_check
import report
import run_eval
import tune_gate

from vskc.dataio import read_jsonl


def run_mock_eval(tmp_path, scenarios_path, knowledge_dir, extra=()):
    results = tmp_path / "results"
    code = run_eval.main(["--split", "custom", "--scenarios", str(scenarios_path),
                          "--knowledge-dir", str(knowledge_dir), "--results-dir", str(results),
                          "--mock", "--run-id", "mock-test"] + list(extra))
    return code, results


def test_eval_report_and_gate(tmp_path, scenarios_path, knowledge_dir):
    code, results = run_mock_eval(tmp_path, scenarios_path, knowledge_dir)
    assert code == 0

    rows = read_jsonl(results / "runs" / "mock-test" / "predictions.jsonl")
    assert len(rows) == 6 * 3
    assert {r["system"] for r in rows} == {"rule", "llm", "rag"}
    assert all(r["retrieved_ids"] == [] for r in rows if r["system"] == "llm")
    assert all(r["top_score"] is None for r in rows if r["system"] == "llm")
    assert any(r["retrieved_ids"] for r in rows if r["system"] == "rag")

    meta = json.loads((results / "runs" / "mock-test" / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["mock"] is True and meta["n_scenarios"] == 6 and len(meta["scenarios_sha256"]) == 64

    raw = read_jsonl(results / "raw" / "mock-test.jsonl")
    assert len(raw) >= 12
    assert all("sk-or-" not in json.dumps(r) for r in raw)
    blob = json.dumps(raw)
    assert "label_rationale" not in blob

    assert tune_gate.main(["--run-id", "mock-test", "--results-dir", str(results),
                           "--max-abstain", "1.0"]) == 0
    gate = json.loads((results / "gate.json").read_text(encoding="utf-8"))
    assert 0 <= gate["tau_conf"] <= 1

    assert report.main(["--results-dir", str(results)]) == 0
    text = (results / "tables" / "report_mock-test.md").read_text(encoding="utf-8")
    for heading in ("T1.", "T2.", "T3.", "T4.", "T5."):
        assert heading in text
    assert "MOCK RUN" in text
    assert "Majority-class baseline" in text


def test_resume_does_not_duplicate(tmp_path, scenarios_path, knowledge_dir):
    code, results = run_mock_eval(tmp_path, scenarios_path, knowledge_dir, ["--limit", "2"])
    assert code == 0
    pred = results / "runs" / "mock-test" / "predictions.jsonl"
    before = len(read_jsonl(pred))
    assert run_eval.main(["--resume", "mock-test", "--results-dir", str(results)]) == 0
    assert len(read_jsonl(pred)) >= before
    keys = [(r["system"], r["scenario_id"]) for r in read_jsonl(pred)]
    assert len(keys) == len(set(keys))


def test_existing_run_id_is_refused(tmp_path, scenarios_path, knowledge_dir):
    assert run_mock_eval(tmp_path, scenarios_path, knowledge_dir)[0] == 0
    assert run_mock_eval(tmp_path, scenarios_path, knowledge_dir)[0] != 0


def test_heldout_lock_detects_earlier_real_run(tmp_path):
    results = tmp_path / "results"
    run_dir = results / "runs" / "heldout-raw-x"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(json.dumps(
        {"run_id": "heldout-raw-x", "split": "heldout", "variant": "raw", "mock": False}), encoding="utf-8")
    mock_dir = results / "runs" / "mock-heldout"
    mock_dir.mkdir(parents=True)
    (mock_dir / "run_meta.json").write_text(json.dumps(
        {"run_id": "mock-heldout", "split": "heldout", "variant": "raw", "mock": True}), encoding="utf-8")
    assert run_eval.previous_heldout_runs(results, "raw") == ["heldout-raw-x"]
    assert run_eval.previous_heldout_runs(results, "stripped") == []


def test_tune_gate_refuses_heldout(tmp_path):
    results = tmp_path / "results"
    run_dir = results / "runs" / "h"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(json.dumps({"run_id": "h", "split": "heldout"}), encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    assert tune_gate.main(["--run-id", "h", "--results-dir", str(results)]) != 0


def test_leakage_script(tmp_path, scenarios_path, knowledge_dir):
    out = tmp_path / "stripped.jsonl"
    code = leakage_check.main(["--in", str(scenarios_path), "--out", str(out),
                               "--knowledge-dir", str(knowledge_dir),
                               "--results-dir", str(tmp_path / "results")])
    assert code == 0
    assert len(read_jsonl(out)) == 6
    assert (tmp_path / "results" / "tables" / "leakage_scenarios_demo.md").exists()
