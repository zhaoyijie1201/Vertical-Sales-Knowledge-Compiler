"""Generate the six scenarios for the FDE configuration-time study.

    python scripts/gen_config_study.py

The study times one person writing a sales-agent instruction for each scenario, three
by hand and three with the system. The scenarios are new: none is in dev or held-out.

The six are fixed by design rather than sampled, so the two groups match in difficulty.
Each group has one scenario whose decision depends on a supplier-side fact, one with a
price objection, and one with another kind of concern. The scenarios are unlabeled:
the study measures time, not accuracy.

Output: data/study/config_study.jsonl. An existing file is never overwritten.
"""
import argparse
import sys
from pathlib import Path

from vskc import paths
from vskc.cli import fail, now_iso, now_stamp, rel, setup_console
from vskc.config import load_settings
from vskc.dataio import append_jsonl
from vskc.generation import SCENARIO_SCHEMA, TEMPLATES, scenario_user_prompt
from vskc.llm import complete_json
from vskc.schema import Scenario

OUT = paths.DATA / "study" / "config_study.jsonl"


def hook(name, side, requirement, refs):
    return {"hook": name, "side": side, "requirement": requirement, "flexibility": "fixed", "fact_refs": refs}


# group, role, attributes
PLAN = [
    ("A", "supplier fact", dict(
        sales_stage="technical_review", customer_size="medium", export_oriented=True, has_incumbent=False,
        objection_type="none", spec_status="given_by_requirement", purchase_timing="within_year",
        length="medium", ambiguous=False,
        hook=hook("temperature", "beyond", "the sensor head has to run continuously at 300 degrees Celsius",
                  ["pd-003"]))),
    ("B", "supplier fact", dict(
        sales_stage="technical_review", customer_size="medium", export_oriented=True, has_incumbent=False,
        objection_type="none", spec_status="given_by_requirement", purchase_timing="within_year",
        length="medium", ambiguous=False,
        hook=hook("viscosity", "beyond",
                  "the process fluid measured by the flow sensor has a viscosity of 4500 millipascal-seconds",
                  ["pd-004"]))),
    ("A", "price objection", dict(
        sales_stage="negotiation", customer_size="large", export_oriented=False, has_incumbent=True,
        objection_type="price", spec_status="confirmed_by_customer", purchase_timing="immediate",
        length="medium", ambiguous=False)),
    ("B", "price objection", dict(
        sales_stage="proposal", customer_size="medium", export_oriented=True, has_incumbent=True,
        objection_type="price", spec_status="confirmed_by_customer", purchase_timing="immediate",
        length="medium", ambiguous=False)),
    ("A", "other concern", dict(
        sales_stage="qualifying", customer_size="small", export_oriented=True, has_incumbent=True,
        objection_type="product_reliability", spec_status="confirmed_by_customer", purchase_timing="within_year",
        length="medium", ambiguous=False)),
    ("B", "other concern", dict(
        sales_stage="technical_review", customer_size="large", export_oriented=False, has_incumbent=True,
        objection_type="industry_experience", spec_status="confirmed_by_customer", purchase_timing="within_year",
        length="medium", ambiguous=False)),
]


def main(argv=None) -> int:
    setup_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="override the generation model; default is the held-out generation model")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    if args.out.exists():
        return fail("%s already exists. Delete it yourself to regenerate." % args.out)

    settings = load_settings(mock=args.mock)
    model = args.model or settings.model_gen_heldout
    if not settings.mock:
        if not model:
            return fail("no generation model configured")
        if model == settings.model_under_test:
            return fail("the generation model must differ from the model under test")

    run_id = "%sgen-study-%s" % ("mock-" if settings.mock else "", now_stamp())
    written = 0
    for i, (group, role, point) in enumerate(PLAN, 1):
        sid = "cs-%02d" % i
        point = dict(point, depends_on_supplier_fact="hook" in point)

        def validate(d, point=point, sid=sid, group=group, role=role):
            return Scenario(
                id=sid, narrative=d["narrative"], sales_stage=point["sales_stage"],
                customer_size=point["customer_size"], export_oriented=point["export_oriented"],
                has_incumbent=point["has_incumbent"],
                pain_points=[str(x) for x in d.get("pain_points", [])][:3],
                objections=[str(x) for x in d.get("objections", [])][:3],
                meta={"source": "synthetic", "purpose": "configuration-time study", "group": group,
                      "role": role, "gen_model": "mock" if settings.mock else model, "template": "B",
                      "grid": point, "generated_at": now_iso(), "gen_run_id": run_id},
            )

        s, _stats, error = complete_json(
            settings=settings, model=model, system=TEMPLATES["B"], user=scenario_user_prompt(point),
            schema=SCENARIO_SCHEMA, schema_name="scenario", validate=validate, run_id=run_id,
            raw_dir=paths.raw_dir(paths.RESULTS),
            tag={"kind": "generation", "scenario_id": sid, "template": "B"},
        )
        if s is None:
            print("%s FAILED: %s" % (sid, error))
            continue
        append_jsonl(args.out, s.model_dump(mode="json"))
        written += 1
        print("%s group %s, %s: %d words" % (sid, group, role, len(s.narrative.split())))
    print("%d written -> %s" % (written, rel(args.out, paths.ROOT)))
    return 0 if written == len(PLAN) else 1


if __name__ == "__main__":
    sys.exit(main())
