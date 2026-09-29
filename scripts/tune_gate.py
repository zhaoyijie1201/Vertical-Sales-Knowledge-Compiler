"""Choose the gate thresholds on a dev run and write results/gate.json.

    python scripts/tune_gate.py --run-id dev-raw-20261002-093000

Selection rule, fixed before looking at any result:
    among threshold pairs whose abstain rate is at most --max-abstain (default 0.25),
    take the one with the highest accuracy on answered cases;
    break ties by the lower abstain rate, then the lower tau_conf, then the lower tau_ret.

Refuses to use a held-out run.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from vskc import paths
from vskc.cli import fail, md_table, now_iso, num, pct, rel, setup_console
from vskc.dataio import read_jsonl
from vskc.metrics import abstention_summary, percentile


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--system", default="rag")
    p.add_argument("--results-dir", type=Path, default=paths.RESULTS)
    p.add_argument("--max-abstain", type=float, default=0.25)
    return p.parse_args(argv)


def main(argv=None) -> int:
    setup_console()
    args = parse_args(argv)
    run_dir = paths.runs_dir(args.results_dir) / args.run_id
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return fail("no such run: %s" % args.run_id)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta["split"] == "heldout":
        return fail("thresholds must not be tuned on held-out")

    rows = [r for r in read_jsonl(run_dir / "predictions.jsonl") if r["system"] == args.system]
    if not rows:
        return fail("run %s has no rows for system %s" % (args.run_id, args.system))

    tau_conf_grid = [round(0.05 * i, 2) for i in range(0, 20)]
    scores = [r["top_score"] for r in rows if r.get("top_score") is not None]
    tau_ret_grid = sorted({0.0} | {round(percentile(scores, q), 4) for q in (0.1, 0.2, 0.3, 0.4)}) \
        if scores else [0.0]

    scan = []
    for tr in tau_ret_grid:
        for tc in tau_conf_grid:
            s = abstention_summary(rows, tc, tr)
            scan.append(s)

    eligible = [s for s in scan if s["abstain_rate"] <= args.max_abstain and s["answered"] > 0]
    if not eligible:
        return fail("no threshold pair keeps the abstain rate at or below %.2f" % args.max_abstain)
    best = sorted(eligible, key=lambda s: (-s["answered_accuracy"], s["abstain_rate"],
                                           s["tau_conf"], s["tau_ret"]))[0]

    gate = {
        "tau_conf": best["tau_conf"],
        "tau_ret": best["tau_ret"],
        "selected_on_run": args.run_id,
        "selected_on_split": meta["split"],
        "system": args.system,
        "rule": "max answered accuracy subject to abstain rate <= %.2f" % args.max_abstain,
        "selected_at": now_iso(),
        "dev_abstain_rate": best["abstain_rate"],
        "dev_answered_accuracy": best["answered_accuracy"],
        "dev_abstained_would_be_wrong": best["abstained_would_be_wrong"],
        "n": best["n"],
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(args.results_dir / "gate.json", "w", encoding="utf-8") as f:
        json.dump(gate, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame([{
        "tau_ret": num(s["tau_ret"], 4),
        "tau_conf": num(s["tau_conf"], 2),
        "abstain rate": pct(s["abstain_rate"]),
        "abstained that would be wrong": pct(s["abstained_would_be_wrong"]),
        "answered accuracy": pct(s["answered_accuracy"]),
        "answered n": s["answered"],
        "selected": "<--" if s is best else "",
    } for s in scan])
    tables = paths.tables_dir(args.results_dir)
    tables.mkdir(parents=True, exist_ok=True)
    out = tables / ("gate_scan_%s.md" % args.run_id)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Gate threshold scan on %s (system: %s, n = %d)\n\n" % (args.run_id, args.system, len(rows)))
        f.write("Selection rule: %s.\n\n" % gate["rule"])
        f.write(md_table(df) + "\n")

    print("selected tau_conf=%.2f tau_ret=%.4f" % (best["tau_conf"], best["tau_ret"]))
    print("  abstain rate              %s" % pct(best["abstain_rate"]))
    print("  abstained would be wrong  %s" % pct(best["abstained_would_be_wrong"]))
    print("  answered accuracy         %s" % pct(best["answered_accuracy"]))
    print("written: %s, %s" % (rel(args.results_dir / "gate.json", paths.ROOT), rel(out, paths.ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
