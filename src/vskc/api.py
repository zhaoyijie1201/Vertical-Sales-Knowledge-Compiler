"""Web interface: a JSON API and a single static page.

    python -m vskc.api                 http://127.0.0.1:8000
    uvicorn vskc.api:app --port 8000

Three ways to get a recommendation:

recorded  replays what each system answered in the evaluation run on disk. No model call,
          no cost, and exactly what was scored.
live      calls the model under test now. Needs the API key.
mock      deterministic fake replies, to show the interface without a key.

The server binds to 127.0.0.1 by default. The API key is never sent to the browser.
"""
import datetime
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, paths
from .config import api_key, load_settings
from .dataio import load_knowledge, load_scenarios, read_jsonl
from .gate import PASS, gate
from .metrics import (abstention_summary, accuracy, cost_per_scenario, majority_baseline,
                      mcnemar_exact, mean, wilson)
from .prompts import PROMPT_VERSION
from .retriever import BM25Retriever
from .schema import ACTION_DEFINITIONS, Action, KnowledgeItem, SalesStage, Scenario
from .systems import SYSTEM_NAMES, SYSTEMS, Context, SystemOutput

WEB_DIR = Path(__file__).resolve().parent / "web"

SYSTEM_LABEL = {"rule": "Rule-based baseline", "llm": "Generic LLM", "rag": "Vertical RAG + LLM"}
GATE_REASON_TEXT = {
    "system_error": "No valid recommendation was produced",
    "cited_unknown_evidence": "Cited evidence that was not retrieved",
    "knowledge_not_covering": "The knowledge base does not cover this case well",
    "low_confidence": "Confidence is below the threshold",
}


# --------------------------------------------------------------------------- request models

class ScenarioIn(BaseModel):
    narrative: str = Field(min_length=1, max_length=6000)
    sales_stage: SalesStage
    customer_size: Literal["small", "medium", "large"]
    export_oriented: bool = False
    has_incumbent: bool = False
    pain_points: List[str] = Field(default_factory=list, max_length=6)
    objections: List[str] = Field(default_factory=list, max_length=6)


class RecommendIn(BaseModel):
    scenario: ScenarioIn
    mode: Literal["live", "mock"] = "mock"
    systems: List[Literal["rule", "llm", "rag"]] = Field(default_factory=lambda: list(SYSTEM_NAMES))


# --------------------------------------------------------------------------- data access

class Store:
    """Reads the repository's files. Small, so everything is read on demand."""

    def __init__(self, results_dir: Path, scenarios_dir: Path, gold_dir: Path, knowledge_dir: Path):
        self.results_dir = Path(results_dir)
        self.scenarios_dir = Path(scenarios_dir)
        self.gold_dir = Path(gold_dir)
        self.knowledge_dir = Path(knowledge_dir)

    # knowledge ------------------------------------------------------------
    def knowledge(self) -> List[KnowledgeItem]:
        return load_knowledge(self.knowledge_dir)

    def retriever(self) -> Optional[BM25Retriever]:
        items = self.knowledge()
        return BM25Retriever(items) if items else None

    # scenarios ------------------------------------------------------------
    def split_path(self, split: str) -> Optional[Path]:
        if split == "heldout":
            p = self.scenarios_dir / "heldout.jsonl"
        elif split == "dev":
            p = self.scenarios_dir / "dev.jsonl"
            if not p.exists():
                p = self.gold_dir / "seed_40.jsonl"
        else:
            return None
        return p if p.exists() else None

    def scenarios(self, split: str) -> List[Scenario]:
        p = self.split_path(split)
        return load_scenarios(p) if p else []

    # runs -----------------------------------------------------------------
    def runs(self) -> List[Dict[str, Any]]:
        root = paths.runs_dir(self.results_dir)
        out = []
        if not root.exists():
            return out
        for d in sorted(root.iterdir()):
            meta_p, pred_p = d / "run_meta.json", d / "predictions.jsonl"
            if not (meta_p.exists() and pred_p.exists()):
                continue
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("mock"):
                continue
            out.append({"meta": meta, "rows": read_jsonl(pred_p)})
        return out

    def latest_run(self, split: str, variant: str = "raw") -> Optional[Dict[str, Any]]:
        cands = [r for r in self.runs() if r["meta"]["split"] == split and r["meta"]["variant"] == variant]
        if not cands:
            return None
        return sorted(cands, key=lambda r: r["meta"]["created_at"])[-1]

    def gate_config(self) -> Optional[Dict[str, Any]]:
        p = self.results_dir / "gate.json"
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)


# --------------------------------------------------------------------------- shaping output

def _item_dict(item: KnowledgeItem, score: Optional[float] = None) -> Dict[str, Any]:
    d = {"id": item.id, "type": item.type, "text": item.text, "source": item.source, "version": item.version}
    if score is not None:
        d["score"] = round(float(score), 2)
    return d


def _gate_dict(system: str, action: Optional[str], confidence: Optional[float], evidence_ids: List[str],
               retrieved_ids: List[str], top_score: Optional[float], gate_cfg: Optional[Dict[str, Any]]
               ) -> Optional[Dict[str, Any]]:
    if system == "rule":
        return None
    if not gate_cfg:
        return {"decision": "unknown", "reason": None,
                "reason_text": "No thresholds have been chosen yet (results/gate.json is missing)"}
    decision, reason = gate(has_recommendation=action is not None, confidence=confidence,
                            evidence_ids=evidence_ids, retrieved_ids=retrieved_ids, top_score=top_score,
                            tau_conf=gate_cfg["tau_conf"], tau_ret=gate_cfg["tau_ret"])
    return {"decision": decision, "reason": reason,
            "reason_text": GATE_REASON_TEXT.get(reason) if reason else "Passes the confidence gate"}


def _cost(input_tokens: int, output_tokens: int, settings: Dict[str, Any]) -> Optional[float]:
    pin, pout = settings.get("price_in_per_mtok"), settings.get("price_out_per_mtok")
    if pin is None or pout is None:
        return None
    return round((input_tokens * pin + output_tokens * pout) / 1e6, 5)


def shape_output(system: str, action: Optional[str], rationale: Optional[str], confidence: Optional[float],
                 evidence_ids: List[str], retrieved: List[Dict[str, Any]], top_score: Optional[float],
                 error: Optional[str], input_tokens: int, output_tokens: int, latency_s: float,
                 by_id: Dict[str, KnowledgeItem], gate_cfg: Optional[Dict[str, Any]],
                 price: Dict[str, Any], gold: Optional[str]) -> Dict[str, Any]:
    retrieved_ids = [r["id"] for r in retrieved]
    evidence = []
    for eid in evidence_ids:
        item = by_id.get(eid)
        evidence.append(dict(_item_dict(item), retrieved=eid in retrieved_ids) if item
                        else {"id": eid, "missing": True, "retrieved": eid in retrieved_ids})
    return {
        "system": system,
        "system_label": SYSTEM_LABEL[system],
        "action": action,
        "rationale": rationale,
        "confidence": confidence,
        "evidence": evidence,
        "retrieved": retrieved,
        "top_score": round(top_score, 2) if top_score is not None else None,
        "gate": _gate_dict(system, action, confidence, evidence_ids, retrieved_ids, top_score, gate_cfg),
        "error": error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_s": latency_s,
        "cost_usd": _cost(input_tokens, output_tokens, price) if system != "rule" else 0.0,
        "correct": (action == gold) if gold else None,
    }


def scenario_dict(s: Scenario, reveal: bool = True) -> Dict[str, Any]:
    m = s.meta
    d = {
        "id": s.id, "narrative": s.narrative, "sales_stage": s.sales_stage.value,
        "customer_size": s.customer_size, "export_oriented": s.export_oriented,
        "has_incumbent": s.has_incumbent, "pain_points": s.pain_points, "objections": s.objections,
        "ambiguous": s.ambiguous,
        "depends_on_supplier_fact": bool(m.get("depends_on_supplier_fact", False)),
        "label_certainty": m.get("label_certainty"),
    }
    if reveal:
        d.update(label=s.label.value if s.label else None, label_rationale=s.label_rationale,
                 alternative_label=m.get("alternative_label"), ambiguity=m.get("ambiguity"))
    return d


# --------------------------------------------------------------------------- results summary

def _acc(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    k, n = accuracy(rows)
    lo, hi = wilson(k, n)
    return {"correct": k, "n": n, "accuracy": k / n if n else None, "ci_low": lo, "ci_high": hi}


def summarize_run(run: Dict[str, Any], fact_ids: set) -> Dict[str, Any]:
    meta, rows = run["meta"], run["rows"]
    by_sys: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_sys.setdefault(r["system"], []).append(r)
    any_rows = next(iter(by_sys.values()))
    label, k, n = majority_baseline([r["gold"] for r in any_rows])

    def dep(r):
        return bool(r.get("depends_on_supplier_fact", r["scenario_id"] in fact_ids))

    slices = {
        "all": lambda r: True,
        "depends_on_supplier_fact": dep,
        "no_supplier_fact": lambda r: not dep(r),
        "clear_label": lambda r: r.get("label_certainty") == "clear",
        "judgment_label": lambda r: r.get("label_certainty") == "judgment",
    }
    out = {
        "run_id": meta["run_id"], "split": meta["split"], "variant": meta["variant"],
        "created_at": meta["created_at"], "model_under_test": meta["settings"].get("model_under_test"),
        "retries": meta.get("retries", []), "n_scenarios": meta["n_scenarios"],
        "majority": {"label": label, **_acc([{"correct": r["gold"] == label} for r in any_rows])},
        "systems": {}, "slices": {}, "paired": [], "gate": None, "cost": {},
    }
    for name in SYSTEM_NAMES:
        if name in by_sys:
            out["systems"][name] = _acc(by_sys[name])
    for sname, keep in slices.items():
        entry = {}
        for name in SYSTEM_NAMES:
            sub = [r for r in by_sys.get(name, []) if keep(r)]
            if sub:
                entry[name] = _acc(sub)
        if entry and next(iter(entry.values()))["n"]:
            out["slices"][sname] = entry

    index = {name: {r["scenario_id"]: r for r in by_sys.get(name, [])} for name in SYSTEM_NAMES}
    for a, b in (("rag", "llm"), ("llm", "rule"), ("rag", "rule")):
        if not (index[a] and index[b]):
            continue
        for sname, keep in slices.items():
            ids = [sid for sid, r in index[a].items() if keep(r) and sid in index[b]]
            if not ids:
                continue
            t = mcnemar_exact([index[a][i]["correct"] for i in ids], [index[b][i]["correct"] for i in ids])
            out["paired"].append({"a": a, "b": b, "slice": sname, **t})

    g = meta.get("gate")
    if g and "rag" in by_sys:
        out["gate"] = dict(abstention_summary(by_sys["rag"], g["tau_conf"], g["tau_ret"]),
                           selected_on=g.get("selected_on_run"), rule=g.get("rule"))
    st = meta["settings"]
    for name in SYSTEM_NAMES:
        rs = by_sys.get(name, [])
        if rs:
            out["cost"][name] = {
                "cost_per_scenario": cost_per_scenario(rs, st.get("price_in_per_mtok"), st.get("price_out_per_mtok"))
                if name != "rule" else 0.0,
                "mean_input_tokens": mean(r["input_tokens"] for r in rs),
                "mean_output_tokens": mean(r["output_tokens"] for r in rs),
                "mean_latency_s": mean(r["latency_s"] for r in rs),
            }
    return out


def compare_variants(store: "Store", raw_run: Dict[str, Any], stripped_run: Dict[str, Any]) -> Dict[str, Any]:
    """Separate the effect of leakage removal from run-to-run variation of the model.

    Only scenarios whose narrative changed can show a leakage effect. Any change on an
    unchanged scenario is variation between two calls with the same input.
    """
    raw_text = {s.id: s.narrative for s in store.scenarios("heldout")}
    p = store.scenarios_dir / "heldout_stripped.jsonl"
    stripped_text = {s.id: s.narrative for s in load_scenarios(p)} if p.exists() else {}
    touched = sorted(i for i in raw_text if i in stripped_text and raw_text[i] != stripped_text[i])
    a = {(r["system"], r["scenario_id"]): r for r in raw_run["rows"]}
    b = {(r["system"], r["scenario_id"]): r for r in stripped_run["rows"]}
    per = {}
    for name in ("llm", "rag"):
        keys = [k for k in a if k[0] == name and k in b]
        if not keys:
            continue
        changed = [k[1] for k in keys if a[k]["action"] != b[k]["action"]]
        flips = [k[1] for k in keys if a[k]["correct"] != b[k]["correct"]]
        per[name] = {"n": len(keys), "action_changed": len(changed),
                     "action_changed_on_untouched": len([i for i in changed if i not in touched]),
                     "correctness_flipped": len(flips),
                     "correctness_flipped_on_touched": len([i for i in flips if i in touched])}
    return {"touched_scenarios": touched, "systems": per}


# --------------------------------------------------------------------------- app

def create_app(results_dir: Path = paths.RESULTS, scenarios_dir: Path = paths.SCENARIOS,
               gold_dir: Path = paths.GOLD, knowledge_dir: Path = paths.KNOWLEDGE) -> FastAPI:
    store = Store(results_dir, scenarios_dir, gold_dir, knowledge_dir)
    app = FastAPI(title="Vertical Sales Knowledge Compiler", version=__version__,
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")

    def fact_ids() -> set:
        p = store.gold_dir / "retrieval_needs.json"
        if not p.exists():
            return set()
        with open(p, "r", encoding="utf-8") as f:
            return set(json.load(f).get("decision_depends_on_supplier_fact", []))

    @app.get("/api/meta")
    def meta() -> Dict[str, Any]:
        st = load_settings()
        try:
            api_key()
            key_ok = True
        except RuntimeError:
            key_ok = False
        return {
            "version": __version__,
            "actions": [{"value": a.value, "when": d["when"], "boundary": d["boundary"]}
                        for a, d in ACTION_DEFINITIONS.items()],
            "stages": [s.value for s in SalesStage],
            "model_under_test": st.model_under_test,
            "retrieval": st.retrieval, "top_k": st.top_k, "prompt_version": PROMPT_VERSION,
            "knowledge_count": len(store.knowledge()),
            "gate": store.gate_config(),
            "live_available": bool(key_ok and st.model_under_test),
            "splits": [s for s in ("heldout", "dev") if store.split_path(s)],
            "gate_reasons": GATE_REASON_TEXT,
        }

    @app.get("/api/scenarios")
    def list_scenarios(split: Literal["heldout", "dev"] = "heldout") -> Dict[str, Any]:
        scen = store.scenarios(split)
        run = store.latest_run(split, "raw")
        recorded: Dict[str, Dict[str, bool]] = {}
        if run:
            for r in run["rows"]:
                recorded.setdefault(r["scenario_id"], {})[r["system"]] = bool(r["correct"])
        ids = fact_ids()
        items = []
        for s in scen:
            d = scenario_dict(s)
            if split == "dev":
                d["depends_on_supplier_fact"] = s.id in ids
            d["snippet"] = s.narrative[:160]
            d["recorded"] = recorded.get(s.id, {})
            del d["narrative"]
            items.append(d)
        return {"split": split, "run_id": run["meta"]["run_id"] if run else None, "items": items}

    @app.get("/api/scenarios/{split}/{sid}")
    def get_scenario(split: Literal["heldout", "dev"], sid: str) -> Dict[str, Any]:
        scen = {s.id: s for s in store.scenarios(split)}
        if sid not in scen:
            raise HTTPException(404, "no scenario %s in %s" % (sid, split))
        s = scen[sid]
        d = scenario_dict(s)
        if split == "dev":
            d["depends_on_supplier_fact"] = s.id in fact_ids()
        run = store.latest_run(split, "raw")
        d["recorded_run"] = None
        if run:
            knowledge = store.knowledge()
            by_id = {k.id: k for k in knowledge}
            retriever = BM25Retriever(knowledge) if knowledge else None
            meta = run["meta"]
            gate_cfg = meta.get("gate") or store.gate_config()
            outs = []
            for r in run["rows"]:
                if r["scenario_id"] != sid:
                    continue
                retrieved = []
                if r["retrieved_ids"]:
                    scores = {}
                    if retriever is not None:
                        mode = meta["settings"].get("retrieval", "single")
                        k = meta["settings"].get("top_k", 5)
                        scores = {it.id: sc for it, sc in retriever.retrieve(s.without_label(), k, mode)}
                    retrieved = [_item_dict(by_id[i], scores.get(i)) if i in by_id else {"id": i, "missing": True}
                                 for i in r["retrieved_ids"]]
                outs.append(shape_output(
                    r["system"], r["action"], r.get("rationale"), r["confidence"], r["evidence_ids"],
                    retrieved, r["top_score"], r["error"], r["input_tokens"], r["output_tokens"],
                    r["latency_s"], by_id, gate_cfg, meta["settings"], r["gold"]))
            order = {n: i for i, n in enumerate(SYSTEM_NAMES)}
            outs.sort(key=lambda o: order[o["system"]])
            d["recorded_run"] = {"run_id": meta["run_id"], "created_at": meta["created_at"],
                                 "model": meta["settings"].get("model_under_test"),
                                 "prompt_version": meta["prompt_version"], "outputs": outs}
        return d

    @app.post("/api/recommend")
    def recommend(body: RecommendIn) -> Dict[str, Any]:
        settings = load_settings(mock=body.mode == "mock")
        if body.mode == "live":
            try:
                api_key()
            except RuntimeError:
                raise HTTPException(400, "Live mode needs an API key. Use recorded or mock mode.")
            if not settings.model_under_test:
                raise HTTPException(400, "No model under test is configured in .env.")
        knowledge = store.knowledge()
        by_id = {k.id: k for k in knowledge}
        retriever = BM25Retriever(knowledge) if knowledge else None
        s = Scenario(id="web", **body.scenario.model_dump())
        run_id = "app-%s" % datetime.date.today().strftime("%Y%m%d")
        if settings.mock:
            run_id = "mock-" + run_id
        ctx = Context(settings=settings, retriever=retriever, run_id=run_id,
                      raw_dir=paths.raw_dir(store.results_dir))
        gate_cfg = store.gate_config()
        outs = []
        for name in body.systems:
            if name == "rag" and retriever is None:
                continue
            try:
                out: SystemOutput = SYSTEMS[name](s, ctx)
            except Exception as e:  # surface service problems to the page, never the key
                raise HTTPException(502, "%s failed: %s" % (SYSTEM_LABEL[name], type(e).__name__))
            rec = out.recommendation
            scores = {}
            if name == "rag" and retriever is not None:
                scores = {it.id: sc for it, sc in retriever.retrieve(s, settings.top_k, settings.retrieval)}
            retrieved = [_item_dict(by_id[i], scores.get(i)) for i in out.retrieved_ids if i in by_id]
            err = out.error
            if err and "Error code: 402" in err:
                err = "The model service has no remaining credit."
            outs.append(shape_output(
                name, rec.action.value if rec else None, rec.rationale if rec else None,
                rec.confidence if rec else None, list(rec.evidence_ids) if rec else [], retrieved,
                out.top_score, err, out.input_tokens, out.output_tokens, out.latency_s, by_id,
                gate_cfg, settings.public_dict(), None))
        return {"mode": body.mode, "model": "mock" if settings.mock else settings.model_under_test,
                "outputs": outs}

    @app.get("/api/knowledge/{kid}")
    def get_knowledge(kid: str) -> Dict[str, Any]:
        for k in store.knowledge():
            if k.id == kid:
                return _item_dict(k)
        raise HTTPException(404, "no knowledge item %s" % kid)

    @app.get("/api/knowledge")
    def list_knowledge() -> Dict[str, Any]:
        return {"items": [_item_dict(k) for k in store.knowledge()]}

    @app.get("/api/results")
    def results() -> Dict[str, Any]:
        ids = fact_ids()
        out = {}
        for key, split, variant in (("heldout_raw", "heldout", "raw"),
                                    ("heldout_stripped", "heldout", "stripped"),
                                    ("dev_raw", "dev", "raw")):
            run = store.latest_run(split, variant)
            if run and run["rows"]:
                n_rows = len(run["rows"])
                expected = run["meta"]["n_scenarios"] * len(run["meta"]["systems"])
                summary = summarize_run(run, ids)
                summary["complete"] = n_rows >= expected and not any(
                    str(r.get("error") or "").startswith("APIStatusError") for r in run["rows"])
                out[key] = summary
        if "heldout_raw" in out and "heldout_stripped" in out:
            out["raw_vs_stripped"] = compare_variants(store, store.latest_run("heldout", "raw"),
                                                      store.latest_run("heldout", "stripped"))
        return out

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("VSKC_HOST", "127.0.0.1")
    port = int(os.environ.get("VSKC_PORT", "8000"))
    print("Vertical Sales Knowledge Compiler: http://%s:%d" % (host, port))
    uvicorn.run("vskc.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
