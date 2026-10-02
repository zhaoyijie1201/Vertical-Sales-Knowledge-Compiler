"""Label scenarios in a spreadsheet.

    python scripts/labeling.py export --split heldout
    python scripts/labeling.py import --split heldout
    python scripts/labeling.py status --split heldout

export  writes labeling/<split>_labels.csv with one row per scenario. The sheet shows what
        a system under test sees, plus the label columns. Generation details are left out
        so they cannot steer the reviewer.
import  reads the sheet back, checks every label against the action set, and writes the
        labels into data/scenarios/<split>.jsonl with the reviewer's name and the date.
        A label that differs from the one on file is recorded as changed in review.
        Narratives and fields are not changed.
status  counts labeled and unlabeled scenarios.

Open the CSV in a spreadsheet program, fill in the columns, and save it as CSV (UTF-8).
Consult data/knowledge/ while labeling: the labeler knows the supplier's products.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from vskc import paths
from vskc.cli import fail, now_iso, rel, setup_console
from vskc.dataio import load_scenarios, write_jsonl
from vskc.schema import ACTION_DEFINITIONS, Action

COLUMNS = ["id", "sales_stage", "customer_size", "export_oriented", "has_incumbent",
           "pain_points", "objections", "narrative", "label", "label_rationale", "ambiguous_note",
           "label_certainty", "alternative_label"]
LABELING_DIR = paths.ROOT / "labeling"


def sheet_path(split: str) -> Path:
    return LABELING_DIR / ("%s_labels.csv" % split)


def data_path(split: str) -> Path:
    return paths.SCENARIOS / ("%s.jsonl" % split)


def cmd_export(args) -> int:
    src, out = data_path(args.split), sheet_path(args.split)
    if not src.exists():
        return fail("not found: %s" % src)
    if out.exists() and not args.force:
        return fail("%s exists and may contain your labels. Use --force to overwrite it." % out)
    scenarios = load_scenarios(src)
    LABELING_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for s in scenarios:
            w.writerow({
                "id": s.id, "sales_stage": s.sales_stage.value, "customer_size": s.customer_size,
                "export_oriented": s.export_oriented, "has_incumbent": s.has_incumbent,
                "pain_points": "; ".join(s.pain_points), "objections": "; ".join(s.objections),
                "narrative": s.narrative,
                "label": s.label.value if s.label else "",
                "label_rationale": s.label_rationale or "",
                "ambiguous_note": s.meta.get("ambiguity", ""),
                "label_certainty": s.meta.get("label_certainty", ""),
                "alternative_label": s.meta.get("alternative_label") or "",
            })
    guide = LABELING_DIR / "actions.txt"
    with open(guide, "w", encoding="utf-8", newline="\n") as f:
        f.write("Allowed values for the label column\n\n")
        for action, d in ACTION_DEFINITIONS.items():
            f.write("%s\n    %s\n    %s\n\n" % (action.value, d["when"], d["boundary"]))
    print("%d scenarios -> %s" % (len(scenarios), rel(out, paths.ROOT)))
    print("action definitions -> %s" % rel(guide, paths.ROOT))
    return 0


def cmd_import(args) -> int:
    src, sheet = data_path(args.split), sheet_path(args.split)
    if not (src.exists() and sheet.exists()):
        return fail("need both %s and %s" % (src, sheet))
    scenarios = load_scenarios(src)
    by_id = {s.id: s for s in scenarios}
    allowed = {a.value for a in Action}

    with open(sheet, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    problems, updates = [], {}
    for n, row in enumerate(rows, 2):
        sid = (row.get("id") or "").strip()
        label = (row.get("label") or "").strip()
        rationale = (row.get("label_rationale") or "").strip()
        if sid not in by_id:
            problems.append("row %d: unknown id %r" % (n, sid))
            continue
        if (row.get("narrative") or "").strip() != by_id[sid].narrative.strip():
            problems.append("row %d (%s): the narrative was edited in the sheet; it must stay as generated" % (n, sid))
        if not label:
            continue
        if label not in allowed:
            problems.append("row %d (%s): %r is not an action" % (n, sid, label))
            continue
        if not rationale:
            problems.append("row %d (%s): label without a rationale" % (n, sid))
            continue
        updates[sid] = (label, rationale, (row.get("ambiguous_note") or "").strip())
    if problems:
        for p in problems:
            print("  " + p)
        return fail("%d problems, nothing was written" % len(problems))

    out = []
    for s in scenarios:
        d = s.model_dump(mode="json")
        if s.id in updates:
            label, rationale, note = updates[s.id]
            d["label"], d["label_rationale"] = label, rationale
            changed = s.label is not None and s.label.value != label
            if changed:
                d["meta"]["label_before_review"] = s.label.value
            d["meta"]["review_status"] = "human_reviewed"
            d["meta"]["reviewed_by"] = args.reviewer
            d["meta"]["reviewed_at"] = now_iso()[:10]
            d["meta"]["changed_in_review"] = changed
            if note:
                d["meta"]["ambiguity"] = note
        out.append(d)
    write_jsonl(src, out)
    labeled = sum(1 for d in out if d["label"])
    changed = sum(1 for d in out if d["meta"].get("changed_in_review"))
    print("%d labels written, %d of %d scenarios now labeled, %d changed in review"
          % (len(updates), labeled, len(out), changed))
    return 0


def cmd_status(args) -> int:
    src = data_path(args.split)
    if not src.exists():
        return fail("not found: %s" % src)
    scenarios = load_scenarios(src)
    labeled = [s for s in scenarios if s.label]
    print("%d of %d labeled" % (len(labeled), len(scenarios)))
    for action, n in sorted(Counter(s.label.value for s in labeled).items()):
        print("  %-26s %d" % (action, n))
    return 0


def main(argv=None) -> int:
    setup_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("export", "import", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("--split", choices=["dev", "heldout"], required=True)
        if name == "export":
            sp.add_argument("--force", action="store_true")
        if name == "import":
            sp.add_argument("--reviewer", required=True, help="name recorded in meta.reviewed_by")
    args = p.parse_args(argv)
    return {"export": cmd_export, "import": cmd_import, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
