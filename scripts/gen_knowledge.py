"""Generate the two synthetic knowledge components: customer SOPs and past outcomes.

    python scripts/gen_knowledge.py --kind sop     --n 30  --seed 11
    python scripts/gen_knowledge.py --kind outcome --n 120 --seed 12

Both are optional. The hand-written playbook and product records are the core knowledge.

Outcomes are sampled from their own grid with their own seed. They are NOT derived from
the evaluation scenarios: an outcome record written about an evaluation scenario would be
retrieved for that same scenario and hand the answer to the model.

Results are balanced across won, lost and stalled, and across customer size and stage,
so the knowledge base does not encode "large customers always win".
"""
import argparse
import sys
from pathlib import Path

from vskc import paths
from vskc.cli import fail, now_iso, now_stamp, rel, setup_console
from vskc.config import load_settings
from vskc.dataio import append_jsonl
from vskc.generation import (OUTCOME_RESULTS, OUTCOME_SYSTEM, SOP_SYSTEM, TEXT_ITEM_SCHEMA,
                             outcome_user_prompt, sample_grid, sop_user_prompt)
from vskc.llm import complete_json
from vskc.schema import KnowledgeItem

KINDS = {
    "sop": {"file": "customer_sops.jsonl", "prefix": "sop", "system": SOP_SYSTEM},
    "outcome": {"file": "outcomes.jsonl", "prefix": "out", "system": OUTCOME_SYSTEM},
}


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=sorted(KINDS), required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", help="override the model slug; defaults to the dev generation model")
    p.add_argument("--knowledge-dir", type=Path, default=paths.KNOWLEDGE)
    p.add_argument("--results-dir", type=Path, default=paths.RESULTS)
    p.add_argument("--mock", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    setup_console()
    args = parse_args(argv)
    kind = KINDS[args.kind]
    out = args.knowledge_dir / kind["file"]
    if out.exists():
        return fail("%s already exists. Delete it yourself if you want to regenerate." % out)

    settings = load_settings(mock=args.mock)
    model = args.model or settings.model_gen_dev
    if not settings.mock and not model:
        return fail("no generation model set in .env")

    run_id = "%sgen-%s-%s" % ("mock-" if settings.mock else "", args.kind, now_stamp())
    points = sample_grid(args.n, args.seed)
    written = failed = 0

    for i, point in enumerate(points, 1):
        kid = "%s-%03d" % (kind["prefix"], i)
        if args.kind == "outcome":
            result = OUTCOME_RESULTS[i % len(OUTCOME_RESULTS)]
            user = outcome_user_prompt(point, result)
        else:
            user = sop_user_prompt(point)

        def validate(d, kid=kid):
            return KnowledgeItem(
                id=kid, type=args.kind, text=d["text"],
                source="synthetic:%s" % ("mock" if settings.mock else model),
                version="seed%d-%s" % (args.seed, now_iso()[:10]),
            )

        item, _stats, error = complete_json(
            settings=settings, model=model, system=kind["system"], user=user,
            schema=TEXT_ITEM_SCHEMA, schema_name="knowledge_text", validate=validate,
            run_id=run_id, raw_dir=paths.raw_dir(args.results_dir),
            tag={"kind": "generation", "knowledge_id": kid, "grid": point},
        )
        if item is None:
            failed += 1
            print("[%d/%d] %s FAILED: %s" % (i, args.n, kid, error))
            continue
        append_jsonl(out, item.model_dump(mode="json"))
        written += 1
        print("[%d/%d] %s ok" % (i, args.n, kid))

    print("%d written, %d failed -> %s" % (written, failed, rel(out, paths.ROOT)))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
