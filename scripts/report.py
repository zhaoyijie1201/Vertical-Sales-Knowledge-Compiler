"""Build every table from files on disk. Makes no model calls and needs no API key.

    python scripts/report.py                     all runs, plus the summary
    python scripts/report.py --run-id RUN_ID     one run

Outputs
    results/tables/report_<run_id>.md   T1 accuracy, T2 abstention, T3 slices, T4 cost, T5 confusion
    results/tables/summary.md           raw against stripped for the latest run of each split
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from vskc import paths
from vskc.cli import fail, md_table, num, pct, rel, setup_console
from vskc.dataio import read_jsonl
from vskc.metrics import (abstention_summary, accuracy, confusion, cost_per_scenario,
                          majority_baseline, mean, percentile, slice_accuracy, wilson)
from vskc.schema import Action

SYSTEM_LABEL = {"rule": "Rule-based baseline", "llm": "Generic LLM (no knowledge)",
                "rag": "Vertical RAG + LLM"}


def load_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path, pred_path = run_dir / "run_meta.json", run_dir / "predictions.jsonl"
    if not (meta_path.exists() and pred_path.exists()):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {"meta": meta, "rows": read_jsonl(pred_path)}


def by_system(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["system"], []).append(r)
    return out


def acc_cell(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return "%s (%d/%d, 95%% CI %s to %s)" % (pct(k / n), k, n, pct(lo), pct(hi))


def t1_accuracy(run) -> pd.DataFrame:
    groups = by_system(run["rows"])
    any_rows = next(iter(groups.values()))
    label, k, n = majority_baseline([r["gold"] for r in any_rows])
    recs = [{"system": "Majority-class baseline (always %s)" % label, "accuracy": acc_cell(k, n),
             "system errors": 0}]
    for name in ("rule", "llm", "rag"):
        if name in groups:
            rows = groups[name]
            k, n = accuracy(rows)
            recs.append({"system": SYSTEM_LABEL[name], "accuracy": acc_cell(k, n),
                         "system errors": sum(1 for r in rows if r["action"] is None)})
    return pd.DataFrame(recs)


def t2_abstention(run, current_gate: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
    """Uses the thresholds recorded when the run was made. A dev run made before any
    thresholds existed falls back to the current results/gate.json and says so."""
    gate = run["meta"].get("gate")
    origin = "fixed before this run"
    if run["meta"]["split"] != "heldout" and current_gate:
        gate, origin = current_gate, "current results/gate.json; thresholds are tuned on dev"
    rows = by_system(run["rows"]).get("rag")
    if not gate or not rows:
        return None
    s = abstention_summary(rows, gate["tau_conf"], gate["tau_ret"])
    k, n = accuracy(rows)
    recs = [
        {"measure": "thresholds (chosen on %s; %s)" % (gate.get("selected_on_run"), origin),
         "value": "tau_conf = %s, tau_ret = %s" % (gate["tau_conf"], gate["tau_ret"])},
        {"measure": "cases", "value": s["n"]},
        {"measure": "abstain rate", "value": "%s (%d/%d)" % (pct(s["abstain_rate"]), s["abstained"], s["n"])},
        {"measure": "abstained cases that would have been wrong",
         "value": "%s (%d/%d)" % (pct(s["abstained_would_be_wrong"]), s["abstained_wrong"], s["abstained"])},
        {"measure": "accuracy on answered cases", "value": acc_cell(s["answered_correct"], s["answered"])},
        {"measure": "accuracy with no gate, for reference", "value": acc_cell(k, n)},
    ]
    for reason, count in sorted(s["reasons"].items()):
        recs.append({"measure": "abstained because: %s" % reason, "value": count})
    return pd.DataFrame(recs)


def t3_slices(run) -> pd.DataFrame:
    groups = by_system(run["rows"])
    recs = []
    for key in ("depends_on_supplier_fact", "label_certainty", "sales_stage", "customer_size",
                "ambiguous", "has_incumbent"):
        if not any(r.get(key) is not None for r in run["rows"]):
            continue
        values = sorted({str(r.get(key)) for r in run["rows"]})
        for v in values:
            rec = {"slice": key, "value": v}
            for name in ("rule", "llm", "rag"):
                if name in groups:
                    k, n = slice_accuracy(groups[name], key).get(v, (0, 0))
                    rec[name] = "%s (%d/%d)" % (pct(k / n), k, n) if n else "n/a"
            recs.append(rec)
    return pd.DataFrame(recs)


def t4_cost(run) -> pd.DataFrame:
    st = run["meta"]["settings"]
    groups = by_system(run["rows"])
    recs = []
    for name in ("rule", "llm", "rag"):
        if name not in groups:
            continue
        rows = groups[name]
        cost = cost_per_scenario(rows, st.get("price_in_per_mtok"), st.get("price_out_per_mtok"))
        lat = [r["latency_s"] for r in rows]
        recs.append({
            "system": SYSTEM_LABEL[name],
            "mean input tokens": num(mean(r["input_tokens"] for r in rows), 0),
            "mean output tokens": num(mean(r["output_tokens"] for r in rows), 0),
            "mean cost per scenario": "n/a (prices not set)" if cost is None else "%.5f" % cost,
            "latency p50 (s)": num(percentile(lat, 0.5)),
            "latency p95 (s)": num(percentile(lat, 0.95)),
            "mean attempts": num(mean(r["attempts"] for r in rows)),
        })
    return pd.DataFrame(recs)


def t5_confusion(run, system: str) -> Optional[pd.DataFrame]:
    rows = by_system(run["rows"]).get(system)
    if not rows:
        return None
    c = confusion(rows)
    labels = [a.value for a in Action]
    cols = labels + (["<error>"] if any(p == "<error>" for _, p in c) else [])
    recs = []
    for g in labels:
        if not any(gg == g for gg, _ in c):
            continue
        rec = {"gold \\ predicted": g}
        for p in cols:
            rec[p] = c.get((g, p), 0)
        recs.append(rec)
    return pd.DataFrame(recs)


def render_run(run, current_gate: Optional[Dict[str, Any]] = None) -> str:
    m = run["meta"]
    st = m["settings"]
    out = ["# Evaluation report: %s" % m["run_id"], ""]
    if m.get("mock"):
        out += ["> MOCK RUN. Replies are deterministic fakes. These numbers test the pipeline "
                "and say nothing about any model.", ""]
    out += [
        "| | |", "|---|---|",
        "| split | %s |" % m["split"],
        "| variant | %s |" % m["variant"],
        "| created | %s |" % m["created_at"],
        "| model under test | %s |" % ("mock" if m.get("mock") else st.get("model_under_test")),
        "| prompt version | %s |" % m["prompt_version"],
        "| top_k | %s |" % st.get("top_k"),
        "| retrieval | %s |" % st.get("retrieval", "single"),
        "| scenarios | %d from `%s` |" % (m["n_scenarios"], m["scenarios_file"]),
        "| knowledge items | %d |" % m["n_knowledge_items"],
        "| scenarios sha256 | `%s` |" % m["scenarios_sha256"][:16],
        "| prices per million tokens | in %s, out %s, checked on %s |"
        % (st.get("price_in_per_mtok"), st.get("price_out_per_mtok"), st.get("price_checked_on") or "n/a"),
    ]
    if m.get("force_reason"):
        out.append("| held-out re-run reason | %s |" % m["force_reason"])
    out.append("")

    out += ["## T1. Next-best-action accuracy", "",
            "Exact match against the gold label. A system error counts as a wrong answer.", "",
            md_table(t1_accuracy(run)), ""]

    t2 = t2_abstention(run, current_gate)
    out += ["## T2. Abstention (Vertical RAG + LLM)", ""]
    out += [md_table(t2), ""] if t2 is not None else \
        ["Not available: no thresholds were recorded for this run. "
         "Run scripts/tune_gate.py on a dev run first.", ""]

    out += ["## T3. Accuracy by slice", "", md_table(t3_slices(run)), ""]
    out += ["## T4. Tokens, cost and latency", "", md_table(t4_cost(run)), ""]

    for name in ("rag", "llm", "rule"):
        t5 = t5_confusion(run, name)
        if t5 is not None:
            out += ["## T5. Confusion matrix: %s" % SYSTEM_LABEL[name], "", md_table(t5), ""]
    return "\n".join(out)


def render_summary(runs: List[Dict[str, Any]]) -> str:
    out = ["# Summary", "",
           "Latest run of each split and variant. Mock runs are excluded.", ""]
    real = [r for r in runs if not r["meta"].get("mock")]
    if not real:
        out.append("No real runs yet.")
        return "\n".join(out)
    for split in ("dev", "heldout", "custom"):
        latest = {}
        for r in sorted(real, key=lambda r: r["meta"]["created_at"]):
            if r["meta"]["split"] == split:
                latest[r["meta"]["variant"]] = r
        if not latest:
            continue
        out += ["## %s" % split, ""]
        recs: Dict[str, Dict[str, Any]] = {}
        for variant in ("raw", "stripped"):
            if variant not in latest:
                continue
            for _, row in t1_accuracy(latest[variant]).iterrows():
                key = row["system"].split(" (always")[0]
                recs.setdefault(key, {"system": key})
                recs[key]["%s (%s)" % (variant, latest[variant]["meta"]["run_id"])] = row["accuracy"]
        out += [md_table(pd.DataFrame(list(recs.values())).fillna("n/a")), ""]
        if len(latest) < 2:
            out += ["Only one variant has been run. Run the other to report the score "
                    "before and after leakage removal.", ""]
    return "\n".join(out)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id")
    p.add_argument("--results-dir", type=Path, default=paths.RESULTS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    setup_console()
    args = parse_args(argv)
    runs_root = paths.runs_dir(args.results_dir)
    if not runs_root.exists():
        return fail("no runs found in %s" % runs_root)
    dirs = [runs_root / args.run_id] if args.run_id else sorted(p for p in runs_root.iterdir() if p.is_dir())
    runs = [r for r in (load_run(d) for d in dirs) if r is not None]
    if not runs:
        return fail("no complete runs found")

    current_gate = None
    gate_path = args.results_dir / "gate.json"
    if gate_path.exists():
        with open(gate_path, "r", encoding="utf-8") as f:
            current_gate = json.load(f)

    tables = paths.tables_dir(args.results_dir)
    tables.mkdir(parents=True, exist_ok=True)
    for run in runs:
        path = tables / ("report_%s.md" % run["meta"]["run_id"])
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(render_run(run, current_gate) + "\n")
        print("written: %s" % rel(path, paths.ROOT))
    if not args.run_id:
        path = tables / "summary.md"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(render_summary(runs) + "\n")
        print("written: %s" % rel(path, paths.ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
