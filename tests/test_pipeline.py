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
                           "--rule", "accuracy", "--max-abstain", "1.0"]) == 0
    gate = json.loads((results / "gate.json").read_text(encoding="utf-8"))
    assert gate["rule_name"] == "accuracy" and 0 <= gate["tau_conf"] <= 1

    assert tune_gate.main(["--run-id", "mock-test", "--results-dir", str(results),
                           "--rule", "percentile", "--percentile", "0.2"]) == 0
    gate = json.loads((results / "gate.json").read_text(encoding="utf-8"))
    assert gate["rule_name"] == "percentile" and 0 < gate["tau_conf"] <= 1
    assert gate["dev_abstain_rate"] <= 0.5

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


def test_labeling_round_trip(tmp_path, scenarios_path, monkeypatch):
    import csv
    import shutil

    import labeling
    from vskc.dataio import load_scenarios, write_jsonl

    data = tmp_path / "scenarios"
    data.mkdir()
    rows = [dict(s.model_dump(mode="json"), label=None, label_rationale=None)
            for s in load_scenarios(scenarios_path)]
    write_jsonl(data / "heldout.jsonl", rows)
    monkeypatch.setattr(labeling.paths, "SCENARIOS", data)
    monkeypatch.setattr(labeling, "LABELING_DIR", tmp_path / "labeling")

    assert labeling.main(["export", "--split", "heldout"]) == 0
    sheet = tmp_path / "labeling" / "heldout_labels.csv"
    with open(sheet, "r", encoding="utf-8-sig", newline="") as f:
        table = list(csv.DictReader(f))
    assert len(table) == 6 and all(r["label"] == "" for r in table)
    assert "grid" not in table[0] and "meta" not in table[0]

    table[0]["label"], table[0]["label_rationale"] = "nurture", "No purchase planned this year."
    table[1]["label"], table[1]["label_rationale"] = "call_them", "x"
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    assert labeling.main(["import", "--split", "heldout", "--reviewer", "tester"]) != 0
    assert all(s.label is None for s in load_scenarios(data / "heldout.jsonl"))

    table[1]["label"], table[1]["label_rationale"] = "", ""
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    assert labeling.main(["import", "--split", "heldout", "--reviewer", "tester"]) == 0
    after = load_scenarios(data / "heldout.jsonl")
    assert after[0].label.value == "nurture" and after[0].meta["reviewed_by"] == "tester"
    assert after[0].meta["review_status"] == "human_reviewed"
    assert [s.narrative for s in after] == [r["narrative"] for r in rows]
    assert sum(1 for s in after if s.label) == 1
