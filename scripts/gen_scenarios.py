"""Generate unlabeled synthetic scenarios from a sampling grid.

    python scripts/gen_scenarios.py --role heldout --n 80 --seed 2026 --template B \
        --out data/scenarios/heldout.jsonl --id-prefix ho

The grid contains no target action and the output has label = null. A person assigns the
gold labels afterwards, before any system is run on the file.

Use a different model and a different template for held-out than for dev, and a model
that is not the model under test.

An existing output file is never overwritten: it may already contain hand-written labels.
"""
import argparse
import sys
from pathlib import Path

from vskc import paths
from vskc.cli import fail, now_iso, now_stamp, rel, setup_console
from vskc.config import load_settings
from vskc.dataio import append_jsonl
from vskc.generation import SCENARIO_SCHEMA, TEMPLATES, sample_plan, scenario_user_prompt
from vskc.llm import complete_json
from vskc.schema import Scenario


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", choices=["dev", "heldout"], required=True,
                   help="which generation model from .env to use")
    p.add_argument("--model", help="override the model slug")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--template", choices=sorted(TEMPLATES), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--id-prefix", required=True)
    p.add_argument("--hook-share", type=float, default=0.0,
                   help="share of scenarios whose decision depends on a supplier-side fact")
    p.add_argument("--results-dir", type=Path, default=paths.RESULTS)
    p.add_argument("--mock", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    setup_console()
    args = parse_args(argv)
    if args.out.exists():
        return fail("%s already exists and may contain labels. Choose another --out." % args.out)

    settings = load_settings(mock=args.mock)
    model = args.model or (settings.model_gen_dev if args.role == "dev" else settings.model_gen_heldout)
    if not settings.mock:
        if not model:
            return fail("no generation model set for role %s in .env" % args.role)
        if model == settings.model_under_test:
            return fail("the generation model must differ from the model under test (%s)" % model)

    run_id = "%sgen-%s-%s" % ("mock-" if settings.mock else "", args.role, now_stamp())
    points = sample_plan(args.n, args.seed, args.hook_share)
    print("%d scenarios, %d with a supplier-fact hook"
          % (len(points), sum(1 for p in points if p["depends_on_supplier_fact"])))
    written = failed = 0

    for i, point in enumerate(points, 1):
        sid = "%s-%03d" % (args.id_prefix, i)

        def validate(d, point=point, sid=sid):
            return Scenario(
                id=sid,
                narrative=d["narrative"],
                sales_stage=point["sales_stage"],
                customer_size=point["customer_size"],
                export_oriented=point["export_oriented"],
                has_incumbent=point["has_incumbent"],
                pain_points=[str(x) for x in d.get("pain_points", [])][:3],
                objections=[str(x) for x in d.get("objections", [])][:3],
                meta={"source": "synthetic", "gen_model": "mock" if settings.mock else model,
                      "template": args.template, "seed": args.seed, "grid": point,
                      "ambiguous": bool(point["ambiguous"]),
                      "depends_on_supplier_fact": bool(point["depends_on_supplier_fact"]),
                      "generated_at": now_iso(),
                      "gen_run_id": run_id},
            )

        scenario, _stats, error = complete_json(
            settings=settings, model=model, system=TEMPLATES[args.template],
            user=scenario_user_prompt(point), schema=SCENARIO_SCHEMA,
            schema_name="scenario", validate=validate, run_id=run_id,
            raw_dir=paths.raw_dir(args.results_dir),
            tag={"kind": "generation", "scenario_id": sid, "template": args.template},
        )
        if scenario is None:
            failed += 1
            print("[%d/%d] %s FAILED: %s" % (i, args.n, sid, error))
            continue
        append_jsonl(args.out, scenario.model_dump(mode="json"))
        written += 1
        print("[%d/%d] %s ok (%d words)" % (i, args.n, sid, len(scenario.narrative.split())))

    print("%d written, %d failed -> %s" % (written, failed, rel(args.out, paths.ROOT)))
    print("next: label every record by hand, then python scripts/validate_data.py")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
