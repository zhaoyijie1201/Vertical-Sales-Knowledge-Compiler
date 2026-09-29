"""Measure retrieval recall on dev. Makes no model call.

    python scripts/eval_retrieval.py

For every dev scenario listed in data/gold/retrieval_needs.json, checks whether at least
one knowledge item that states the needed supplier fact is among the top-k retrieved
items, for each retrieval mode and several values of k.

Output: results/tables/retrieval_recall.md
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from vskc import paths
from vskc.cli import fail, md_table, rel, setup_console
from vskc.dataio import load_knowledge, load_scenarios
from vskc.retriever import RETRIEVAL_MODES, BM25Retriever

KS = (3, 5, 8, 10)


def main(argv=None) -> int:
    setup_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--needs", type=Path, default=paths.GOLD / "retrieval_needs.json")
    p.add_argument("--scenarios", type=Path, default=paths.GOLD / "seed_40.jsonl")
    p.add_argument("--knowledge-dir", type=Path, default=paths.KNOWLEDGE)
    p.add_argument("--results-dir", type=Path, default=paths.RESULTS)
    args = p.parse_args(argv)

    if not args.needs.exists():
        return fail("not found: %s" % args.needs)
    with open(args.needs, "r", encoding="utf-8") as f:
        needs = json.load(f)["needs"]
    scenarios = {s.id: s.without_label() for s in load_scenarios(args.scenarios)}
    knowledge = load_knowledge(args.knowledge_dir)
    known = {k.id for k in knowledge}
    missing = sorted({i for ids in needs.values() for i in ids} - known)
    if missing:
        return fail("retrieval_needs.json names unknown knowledge items: %s" % missing)
    retriever = BM25Retriever(knowledge)

    recall, ranks = [], []
    for mode in RETRIEVAL_MODES:
        row = {"retrieval": mode}
        full = {sid: [h[0].id for h in retriever.retrieve(scenarios[sid], len(knowledge), mode)]
                for sid in needs}
        for k in KS:
            hit = sum(any(i in full[sid][:k] for i in ids) for sid, ids in needs.items())
            row["k=%d" % k] = "%d/%d (%.0f%%)" % (hit, len(needs), 100.0 * hit / len(needs))
        recall.append(row)
        for sid, ids in sorted(needs.items()):
            found = [full[sid].index(i) + 1 for i in ids if i in full[sid]]
            ranks.append({"retrieval": mode, "scenario": sid, "needed": ", ".join(ids),
                          "best rank": min(found) if found else "not retrieved"})

    tables = paths.tables_dir(args.results_dir)
    tables.mkdir(parents=True, exist_ok=True)
    out = tables / "retrieval_recall.md"
    rank_df = pd.DataFrame(ranks).pivot(index=["scenario", "needed"], columns="retrieval",
                                        values="best rank").reset_index()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Retrieval recall on dev\n\n")
        f.write("%d scenarios need a supplier-side fact; %d knowledge items.\n\n" % (len(needs), len(knowledge)))
        f.write("`single` ranks one query built from the whole scenario. `fused` ranks the narrative "
                "and the pain point and objection phrases separately and merges the two rankings "
                "by reciprocal rank fusion.\n\n")
        f.write(md_table(pd.DataFrame(recall)) + "\n\n")
        f.write("## Rank of the needed item\n\n")
        f.write(md_table(rank_df) + "\n\n")
        f.write("The retrieval mode and k were chosen on these %d dev scenarios, so the recall shown "
                "here is optimistic for unseen scenarios.\n" % len(needs))

    print(md_table(pd.DataFrame(recall)))
    print("written: %s" % rel(out, paths.ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
