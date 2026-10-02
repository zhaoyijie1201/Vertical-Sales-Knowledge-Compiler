"""Check the data files against the schema and the labeling quotas.

    python scripts/validate_data.py            everything that exists
    python scripts/validate_data.py --gold     only data/gold/seed_40.jsonl

Exit code 0 when every check passes, 1 otherwise. Warnings do not fail the run.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List

from vskc import paths
from vskc.cli import setup_console
from vskc.dataio import load_knowledge, load_scenarios
from vskc.leakage import find_leaks
from vskc.metrics import majority_baseline
from vskc.schema import GOLD_ACTIONS, SalesStage, Scenario

QUOTA = {
    "total": 40,
    "per_action_min": 3, "per_action_max": 8,
    "per_stage_min": 5,
    "per_size_min": 10,
    "incumbent_each_min": 12,
    "ambiguous_min": 6,
    "short_min": 8, "short_words": 100,
    "long_min": 8, "long_words": 160,
}


class Checker:
    def __init__(self):
        self.failed = 0
        self.warned = 0

    def check(self, ok: bool, message: str) -> None:
        print("  [%s] %s" % ("ok" if ok else "FAIL", message))
        self.failed += int(not ok)

    def warn(self, message: str) -> None:
        print("  [warn] %s" % message)
        self.warned += 1


def words(s: Scenario) -> int:
    return len(s.narrative.split())


def check_gold(path: Path, c: Checker) -> None:
    print("gold labels: %s" % path)
    if not path.exists():
        c.check(False, "file does not exist yet")
        return
    try:
        rows: List[Scenario] = load_scenarios(path)
    except ValueError as e:
        c.check(False, str(e))
        return
    c.check(True, "every record matches the schema, ids are unique")
    q = QUOTA
    c.check(len(rows) == q["total"], "%d records (need %d)" % (len(rows), q["total"]))
    c.check(all(r.label is not None for r in rows), "every record has a label")
    c.check(all(r.label_rationale for r in rows), "every record has a label_rationale")
    expected = {"gold-%03d" % i for i in range(1, q["total"] + 1)}
    c.check({r.id for r in rows} == expected, "ids are gold-001 to gold-%03d" % q["total"])

    actions = Counter(r.label.value for r in rows if r.label)
    for a in GOLD_ACTIONS:
        n = actions.get(a.value, 0)
        c.check(q["per_action_min"] <= n <= q["per_action_max"],
                "action %-26s %2d (need %d to %d)" % (a.value, n, q["per_action_min"], q["per_action_max"]))
    stages = Counter(r.sales_stage.value for r in rows)
    for st in SalesStage:
        c.check(stages.get(st.value, 0) >= q["per_stage_min"],
                "stage %-18s %2d (need at least %d)" % (st.value, stages.get(st.value, 0), q["per_stage_min"]))
    sizes = Counter(r.customer_size for r in rows)
    for size in ("small", "medium", "large"):
        c.check(sizes.get(size, 0) >= q["per_size_min"],
                "size %-8s %2d (need at least %d)" % (size, sizes.get(size, 0), q["per_size_min"]))
    inc = Counter(r.has_incumbent for r in rows)
    c.check(inc.get(True, 0) >= q["incumbent_each_min"] and inc.get(False, 0) >= q["incumbent_each_min"],
            "has_incumbent true %d, false %d (need at least %d each)"
            % (inc.get(True, 0), inc.get(False, 0), q["incumbent_each_min"]))
    amb = sum(1 for r in rows if r.ambiguous)
    c.check(amb >= q["ambiguous_min"], "ambiguous cases %d (need at least %d)" % (amb, q["ambiguous_min"]))
    short = sum(1 for r in rows if words(r) < q["short_words"])
    long_ = sum(1 for r in rows if words(r) > q["long_words"])
    c.check(short >= q["short_min"], "narratives under %d words: %d (need at least %d)"
            % (q["short_words"], short, q["short_min"]))
    c.check(long_ >= q["long_min"], "narratives over %d words: %d (need at least %d)"
            % (q["long_words"], long_, q["long_min"]))

    pending: List[str] = []
    for r in rows:
        hits = find_leaks(r.narrative)
        if hits:
            c.warn("%s narrative contains a leak term: %s" % (r.id, ", ".join(m for _, m in hits)))
        if r.ambiguous and not r.meta.get("ambiguity"):
            c.warn("%s is marked ambiguous but meta.ambiguity does not say why" % r.id)
        if "pending" in str(r.meta.get("review_status", "")):
            pending.append(r.id)
    if pending:
        c.warn("%d records still await human review (meta.review_status): %s to %s"
               % (len(pending), pending[0], pending[-1]))

    if actions:
        label, n, _ = majority_baseline([r.label.value for r in rows if r.label])
        print("  majority-class baseline on this file: always '%s' scores %d/%d = %.1f%%"
              % (label, n, len(rows), 100.0 * n / len(rows)))


def check_scenarios(path: Path, c: Checker) -> None:
    print("scenarios: %s" % path)
    try:
        rows = load_scenarios(path)
    except ValueError as e:
        c.check(False, str(e))
        return
    c.check(True, "%d records match the schema, ids are unique" % len(rows))
    unlabeled = sum(1 for r in rows if r.label is None)
    if unlabeled:
        c.warn("%d records have no label yet" % unlabeled)


def check_knowledge(directory: Path, c: Checker) -> None:
    print("knowledge base: %s" % directory)
    try:
        items = load_knowledge(directory)
    except ValueError as e:
        c.check(False, str(e))
        return
    if not items:
        c.warn("no knowledge items yet")
        return
    c.check(True, "%d items match the schema, ids are unique" % len(items))
    for kind, n in sorted(Counter(i.type for i in items).items()):
        print("  %-9s %d" % (kind, n))
    for i in items:
        hits = [h for h in find_leaks(i.text) if h[0] == "advice"]
        if hits:
            c.warn("%s reads as advice for one case (%s); state a general principle instead"
                   % (i.id, hits[0][1]))


def main(argv=None) -> int:
    setup_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", action="store_true", help="only check the gold file")
    args = p.parse_args(argv)

    c = Checker()
    check_gold(paths.GOLD / "seed_40.jsonl", c)
    if not args.gold:
        print()
        check_knowledge(paths.KNOWLEDGE, c)
        if paths.SCENARIOS.exists():
            for path in sorted(paths.SCENARIOS.glob("*.jsonl")):
                print()
                check_scenarios(path, c)
    print()
    print("%d failed, %d warnings" % (c.failed, c.warned))
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
