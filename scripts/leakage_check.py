"""Find and remove leakage, and write the "stripped" variant of a scenario file.

    python scripts/leakage_check.py --split dev
    python scripts/leakage_check.py --split heldout
    python scripts/leakage_check.py --in path/to/file.jsonl --out path/to/file_stripped.jsonl

Also lists knowledge items whose wording overlaps heavily with a gold rationale.

Outputs
    data/scenarios/<split>_stripped.jsonl
    results/tables/leakage_<split>.md
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from vskc import paths
from vskc.cli import fail, md_table, pct, rel, setup_console
from vskc.dataio import load_knowledge, load_scenarios, write_jsonl
from vskc.leakage import PLACEHOLDER, knowledge_rationale_overlap, strip_scenario


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["dev", "heldout"])
    p.add_argument("--in", dest="inp", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--knowledge-dir", type=Path, default=paths.KNOWLEDGE)
    p.add_argument("--results-dir", type=Path, default=paths.RESULTS)
    p.add_argument("--overlap-threshold", type=float, default=0.5)
    return p.parse_args(argv)


def main(argv=None) -> int:
    setup_console()
    args = parse_args(argv)
    if args.split:
        inp = paths.SCENARIOS / ("%s.jsonl" % args.split)
        if args.split == "dev" and not inp.exists():
            inp = paths.GOLD / "seed_40.jsonl"
        out = paths.SCENARIOS / ("%s_stripped.jsonl" % args.split)
        name = args.split
    elif args.inp and args.out:
        inp, out, name = args.inp, args.out, args.inp.stem
    else:
        return fail("give --split, or both --in and --out")
    if not inp.exists():
        return fail("scenario file not found: %s" % inp)

    scenarios = load_scenarios(inp)
    stripped, hit_rows, emptied = [], [], 0
    for s in scenarios:
        new, hits = strip_scenario(s)
        stripped.append(new.model_dump(mode="json"))
        emptied += int(new.narrative == PLACEHOLDER)
        for h in hits:
            hit_rows.append({"scenario": s.id, "category": h["category"],
                             "match": h["match"], "sentence removed": h["sentence"]})
    write_jsonl(out, stripped)

    affected = len({h["scenario"] for h in hit_rows})
    knowledge = load_knowledge(args.knowledge_dir)
    overlaps = knowledge_rationale_overlap(knowledge, scenarios, args.overlap_threshold)

    tables = paths.tables_dir(args.results_dir)
    tables.mkdir(parents=True, exist_ok=True)
    report = tables / ("leakage_%s.md" % name)
    with open(report, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Leakage check: %s\n\n" % name)
        f.write("Source: `%s`, %d scenarios.\n\n" % (rel(inp, paths.ROOT), len(scenarios)))
        f.write("- Scenarios with at least one removed sentence: %d (%s)\n"
                % (affected, pct(affected / len(scenarios)) if scenarios else "n/a"))
        f.write("- Sentences removed: %d\n" % len(hit_rows))
        f.write("- Scenarios left with no text: %d\n\n" % emptied)
        f.write("## Removed sentences\n\n")
        f.write((md_table(pd.DataFrame(hit_rows)) if hit_rows else "None.") + "\n\n")
        f.write("## Knowledge items close to a gold rationale\n\n")
        f.write("Token-set Jaccard at or above %.2f. Review each one by hand: a knowledge item "
                "should state a general principle, not restate the answer to one scenario.\n\n"
                % args.overlap_threshold)
        f.write((md_table(pd.DataFrame(overlaps)) if overlaps else "None.") + "\n")

    print("%d of %d scenarios had sentences removed (%d sentences)" % (affected, len(scenarios), len(hit_rows)))
    print("%d knowledge items overlap with a gold rationale" % len(overlaps))
    print("written: %s, %s" % (rel(out, paths.ROOT), rel(report, paths.ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
