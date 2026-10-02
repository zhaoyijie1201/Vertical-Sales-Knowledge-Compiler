"""Run the systems on a split and write one prediction row per (system, scenario).

    python scripts/run_eval.py --split dev
    python scripts/run_eval.py --split dev --variant stripped
    python scripts/run_eval.py --split heldout
    python scripts/run_eval.py --resume heldout-raw-20261003-101500
    python scripts/run_eval.py --resume heldout-raw-20261003-101500 --retry-errors

Outputs
    results/runs/<run_id>/predictions.jsonl   one row per (system, scenario)
    results/runs/<run_id>/run_meta.json       models, prompt version, thresholds, data hashes
    results/raw/<run_id>.jsonl                every model call, including failed attempts

The held-out split is run once per variant. A second run is refused unless --force is
given with --reason, and the reason is recorded. Thresholds must exist before held-out
is run, so they cannot be chosen after seeing held-out results.

A run stops at the first non-transient service error (no credit, invalid key, no access):
continuing would only write rows without a recommendation. --resume continues the run.
--retry-errors also re-runs the rows that failed with a service error. Rows where the
model replied but the reply failed validation are kept: those are system errors and count
as wrong answers. A resume refuses to continue when the scenarios, the knowledge base,
the prompt or the model differ from the original run.
"""
import re
import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import vskc
from vskc import paths
from vskc.cli import fail, now_iso, now_stamp, rel, setup_console
from vskc.config import load_settings
from vskc.dataio import append_jsonl, load_knowledge, load_scenarios, read_jsonl, sha256_dir, sha256_file
from vskc.prompts import PROMPT_VERSION
from vskc.retriever import BM25Retriever
from vskc.schema import Scenario
from vskc.systems import SYSTEM_NAMES, SYSTEMS, Context, SystemOutput

# Errors raised by the model provider's service, as opposed to a reply that failed validation.
SERVICE_ERROR = re.compile(
    r"^(APIStatusError|APIConnectionError|APITimeoutError|RateLimitError|InternalServerError|"
    r"AuthenticationError|PermissionDeniedError|BadRequestError|NotFoundError|ConflictError|"
    r"UnprocessableEntityError)\b")
# Service errors that will not go away by retrying within the same run.
FATAL_SERVICE_ERROR = re.compile(r"Error code: (401|402|403)\b")


def is_service_error(error: Optional[str]) -> bool:
    return bool(error) and bool(SERVICE_ERROR.match(error))


def default_scenarios_path(split: str, variant: str) -> Path:
    name = split if variant == "raw" else "%s_stripped" % split
    path = paths.SCENARIOS / ("%s.jsonl" % name)
    if split == "dev" and variant == "raw" and not path.exists():
        gold = paths.GOLD / "seed_40.jsonl"
        if gold.exists():
            print("note: data/scenarios/dev.jsonl not found, using data/gold/seed_40.jsonl as dev")
            return gold
    return path


def load_gate(results_dir: Path) -> Optional[Dict[str, Any]]:
    path = results_dir / "gate.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def previous_heldout_runs(results_dir: Path, variant: str) -> List[str]:
    found = []
    runs = paths.runs_dir(results_dir)
    if not runs.exists():
        return found
    for meta_path in sorted(runs.glob("*/run_meta.json")):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("split") == "heldout" and meta.get("variant") == variant and not meta.get("mock"):
            found.append(meta["run_id"])
    return found


def fact_dependent_ids() -> Set[str]:
    """Dev scenarios flagged in data/gold/retrieval_needs.json. Generated scenarios carry
    the flag in their own meta."""
    path = paths.GOLD / "retrieval_needs.json"
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f).get("decision_depends_on_supplier_fact", []))


def to_row(out: SystemOutput, s: Scenario, run_id: str, fact_ids: Set[str] = frozenset()) -> Dict[str, Any]:
    rec = out.recommendation
    action = rec.action.value if rec is not None else None
    gold = s.label.value
    evidence_ids = list(rec.evidence_ids) if rec is not None else []
    return {
        "run_id": run_id,
        "system": out.system,
        "scenario_id": s.id,
        "gold": gold,
        "action": action,
        "correct": action == gold,
        "confidence": rec.confidence if rec is not None else None,
        "rationale": rec.rationale if rec is not None else None,
        "evidence_ids": evidence_ids,
        "retrieved_ids": out.retrieved_ids,
        "evidence_valid": set(evidence_ids) <= set(out.retrieved_ids),
        "top_score": out.top_score,
        "error": out.error,
        "attempts": out.attempts,
        "input_tokens": out.input_tokens,
        "output_tokens": out.output_tokens,
        "latency_s": out.latency_s,
        "sales_stage": s.sales_stage.value,
        "customer_size": s.customer_size,
        "has_incumbent": s.has_incumbent,
        "export_oriented": s.export_oriented,
        "ambiguous": s.ambiguous,
        "depends_on_supplier_fact": bool(s.meta.get("depends_on_supplier_fact", s.id in fact_ids)),
        "label_certainty": s.meta.get("label_certainty"),
    }


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["dev", "heldout", "custom"], default="dev")
    p.add_argument("--variant", choices=["raw", "stripped"], default="raw")
    p.add_argument("--scenarios", type=Path, help="scenario file; required for --split custom")
    p.add_argument("--knowledge-dir", type=Path, default=paths.KNOWLEDGE)
    p.add_argument("--results-dir", type=Path, default=paths.RESULTS)
    p.add_argument("--systems", default=",".join(SYSTEM_NAMES))
    p.add_argument("--limit", type=int, help="only the first N scenarios (dev and custom only)")
    p.add_argument("--run-id")
    p.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    p.add_argument("--retry-errors", action="store_true",
                   help="with --resume: also re-run rows that failed with a service error")
    p.add_argument("--mock", action="store_true", help="no model calls; deterministic fake replies")
    p.add_argument("--force", action="store_true", help="allow a second held-out run")
    p.add_argument("--reason", help="why held-out is being run again; recorded in run_meta.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    setup_console()
    args = parse_args(argv)
    results_dir: Path = args.results_dir
    runs = paths.runs_dir(results_dir)

    resumed_meta: Optional[Dict[str, Any]] = None
    if args.resume:
        meta_path = runs / args.resume / "run_meta.json"
        if not meta_path.exists():
            return fail("no such run to resume: %s" % args.resume)
        with open(meta_path, "r", encoding="utf-8") as f:
            resumed_meta = json.load(f)
        args.split, args.variant = resumed_meta["split"], resumed_meta["variant"]
        args.systems = ",".join(resumed_meta["systems"])
        args.mock = bool(resumed_meta["mock"])
        args.limit = resumed_meta.get("limit")
        args.knowledge_dir = paths.ROOT / resumed_meta["knowledge_dir"]
        args.scenarios = paths.ROOT / resumed_meta["scenarios_file"]
        if not args.scenarios.exists():
            args.scenarios = Path(resumed_meta["scenarios_file"])
    elif args.retry_errors:
        return fail("--retry-errors needs --resume")

    systems = [x.strip() for x in args.systems.split(",") if x.strip()]
    unknown = [x for x in systems if x not in SYSTEMS]
    if unknown:
        return fail("unknown systems: %s" % unknown)

    settings = load_settings(mock=args.mock)

    if args.split == "custom" and not args.scenarios:
        return fail("--split custom needs --scenarios")
    scenarios_path = args.scenarios or default_scenarios_path(args.split, args.variant)
    if not scenarios_path.exists():
        return fail("scenario file not found: %s" % scenarios_path)

    if args.split == "heldout" and args.limit:
        return fail("--limit is not allowed on held-out")

    gate_cfg = load_gate(results_dir)
    force_reason = None
    if args.split == "heldout" and not settings.mock and resumed_meta is None:
        if gate_cfg is None:
            return fail("results/gate.json not found. Choose thresholds on dev with "
                        "scripts/tune_gate.py before running held-out.")
        earlier = previous_heldout_runs(results_dir, args.variant)
        if earlier:
            if not (args.force and args.reason):
                return fail("held-out (%s) was already run: %s. It is run once. "
                            "Use --resume to continue an interrupted run, or "
                            "--force --reason \"...\" to run again and record why."
                            % (args.variant, ", ".join(earlier)))
            force_reason = args.reason

    all_scenarios = load_scenarios(scenarios_path)
    labeled = [s for s in all_scenarios if s.label is not None]
    skipped = len(all_scenarios) - len(labeled)
    if skipped:
        print("warning: %d scenarios have no label and are skipped" % skipped)
    if args.limit:
        labeled = labeled[: args.limit]
    if not labeled:
        return fail("no labeled scenarios in %s" % scenarios_path)

    knowledge = load_knowledge(args.knowledge_dir)
    if "rag" in systems and not knowledge:
        return fail("the rag system needs knowledge items, none found in %s" % args.knowledge_dir)
    retriever = BM25Retriever(knowledge) if knowledge else None

    if not settings.mock and any(x in systems for x in ("llm", "rag")) and not settings.model_under_test:
        return fail("VSKC_MODEL_UNDER_TEST is not set in .env")

    if resumed_meta is not None:
        run_id = resumed_meta["run_id"]
    else:
        run_id = args.run_id or "%s%s-%s-%s" % ("mock-" if settings.mock else "", args.split,
                                                args.variant, now_stamp())
    run_dir = runs / run_id
    pred_path = run_dir / "predictions.jsonl"
    if resumed_meta is None and run_dir.exists():
        return fail("run directory already exists: %s" % run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if resumed_meta is not None:
        current = {
            "scenarios_sha256": sha256_file(scenarios_path),
            "knowledge_sha256": sha256_dir(args.knowledge_dir),
            "prompt_version": PROMPT_VERSION,
            "model_under_test": settings.model_under_test,
            "top_k": settings.top_k,
            "retrieval": settings.retrieval,
        }
        recorded = {
            "scenarios_sha256": resumed_meta["scenarios_sha256"],
            "knowledge_sha256": resumed_meta["knowledge_sha256"],
            "prompt_version": resumed_meta["prompt_version"],
            "model_under_test": resumed_meta["settings"].get("model_under_test"),
            "top_k": resumed_meta["settings"].get("top_k"),
            "retrieval": resumed_meta["settings"].get("retrieval", "single"),
        }
        if settings.mock:
            current.pop("model_under_test"), recorded.pop("model_under_test")
        changed = sorted(k for k in current if current[k] != recorded[k])
        if changed:
            return fail("cannot resume %s: %s changed since the run started"
                        % (resumed_meta["run_id"], ", ".join(changed)))

    existing = read_jsonl(pred_path) if pred_path.exists() else []
    if args.retry_errors and existing:
        retry = [r for r in existing if is_service_error(r.get("error"))]
        if retry:
            keep = [r for r in existing if not is_service_error(r.get("error"))]
            with open(pred_path, "w", encoding="utf-8", newline="\n") as f:
                for r in keep:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in retry:
                append_jsonl(run_dir / "retried_rows.jsonl", dict(r, retried_at=now_iso()))
            kinds = sorted({r["error"].split(" - ")[0][:60] for r in retry})
            resumed_meta.setdefault("retries", []).append({
                "at": now_iso(), "rows": len(retry), "errors": kinds,
                "note": "rows without a model reply because of a service error, re-run with "
                        "unchanged scenarios, knowledge, prompt, model and thresholds",
            })
            with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
                json.dump(resumed_meta, f, ensure_ascii=False, indent=2)
            existing = keep
            print("retrying %d rows that failed with a service error" % len(retry))
    done: Set[Tuple[str, str]] = {(r["system"], r["scenario_id"]) for r in existing}

    if resumed_meta is None:
        meta = {
            "run_id": run_id,
            "created_at": now_iso(),
            "split": args.split,
            "variant": args.variant,
            "systems": systems,
            "mock": settings.mock,
            "settings": settings.public_dict(),
            "prompt_version": PROMPT_VERSION,
            "scenarios_file": rel(scenarios_path, paths.ROOT),
            "scenarios_sha256": sha256_file(scenarios_path),
            "knowledge_dir": rel(args.knowledge_dir, paths.ROOT),
            "knowledge_sha256": sha256_dir(args.knowledge_dir),
            "n_knowledge_items": len(knowledge),
            "n_scenarios": len(labeled),
            "limit": args.limit,
            "n_unlabeled_skipped": skipped,
            "gate": gate_cfg,
            "force_reason": force_reason,
            "vskc_version": vskc.__version__,
            "python": platform.python_version(),
        }
        with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    fact_ids = fact_dependent_ids()
    ctx = Context(settings=settings, retriever=retriever, run_id=run_id,
                  raw_dir=paths.raw_dir(results_dir))

    print("run %s: %d scenarios x %s%s" % (run_id, len(labeled), systems,
                                           " [mock]" if settings.mock else ""))
    total = len(labeled) * len(systems)
    n = 0
    for s in labeled:
        for name in systems:
            n += 1
            if (name, s.id) in done:
                continue
            out = SYSTEMS[name](s, ctx)
            if out.error and FATAL_SERVICE_ERROR.search(out.error):
                print("stopped at [%d/%d] %s %s: %s" % (n, total, name, s.id, out.error[:160]))
                return fail("the model service refused the request. Fix the account, then run:\n"
                            "  python scripts/run_eval.py --resume %s" % run_id, code=3)
            row = to_row(out, s, run_id, fact_ids)
            append_jsonl(pred_path, row)
            mark = "ok " if row["correct"] else ("ERR" if row["action"] is None else "x  ")
            print("[%d/%d] %-4s %-12s %s gold=%s pred=%s"
                  % (n, total, name, s.id, mark, row["gold"], row["action"]))

    print("predictions: %s" % rel(pred_path, paths.ROOT))
    print("next: python scripts/report.py --run-id %s" % run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
