"""Streamlit demo: one scenario, three systems side by side.

    streamlit run src/vskc/app.py

Decision support only. Nothing here contacts a customer or writes to any external system.
"""
import datetime
import json

import streamlit as st

from vskc import paths
from vskc.config import load_settings
from vskc.dataio import load_knowledge
from vskc.gate import PASS, gate
from vskc.retriever import BM25Retriever
from vskc.schema import ACTION_DEFINITIONS, SalesStage, Scenario
from vskc.systems import SYSTEMS, Context

LABELS = {"rule": "Rule-based baseline", "llm": "Generic LLM (no knowledge)",
          "rag": "Vertical RAG + LLM"}

EXAMPLES = {
    "Exporter evaluating alternative suppliers": dict(
        narrative=("An export-oriented industrial customer has an existing supplier, has "
                   "experienced long lead times, and is evaluating alternative industrial "
                   "sensor suppliers."),
        sales_stage="qualifying", customer_size="medium", export_oriented=True,
        has_incumbent=True, pain_points="long lead time", objections=""),
    "Blank": dict(narrative="", sales_stage="prospecting", customer_size="medium",
                  export_oriented=False, has_incumbent=False, pain_points="", objections=""),
}


@st.cache_resource
def load_kb():
    items = load_knowledge(paths.KNOWLEDGE)
    return items, (BM25Retriever(items) if items else None)


def load_gate():
    path = paths.RESULTS / "gate.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def lines(text: str):
    return [x.strip() for x in text.replace(";", "\n").split("\n") if x.strip()]


def main():
    st.set_page_config(page_title="Vertical Sales Knowledge Compiler", layout="wide")
    st.title("Vertical Sales Knowledge Compiler")
    st.caption("Next-best-action decision support for a Forward-Deployed Engineer. "
               "Recommendations are suggestions for human review, not instructions.")

    items, retriever = load_kb()
    by_id = {i.id: i for i in items}
    gate_cfg = load_gate()

    with st.sidebar:
        st.header("Settings")
        mock = st.toggle("Mock mode (no model call)", value=False)
        settings = load_settings(mock=mock)
        st.write("Model under test:", "mock" if settings.mock else (settings.model_under_test or "not set"))
        st.write("Knowledge items:", len(items))
        st.write("top_k:", settings.top_k)
        default_conf = float(gate_cfg["tau_conf"]) if gate_cfg else 0.6
        default_ret = float(gate_cfg["tau_ret"]) if gate_cfg else 0.0
        tau_conf = st.slider("Confidence threshold", 0.0, 1.0, default_conf, 0.05)
        tau_ret = st.number_input("Retrieval score threshold", min_value=0.0, value=default_ret, step=0.5)
        st.caption("Thresholds come from results/gate.json, chosen on the dev split."
                   if gate_cfg else "No results/gate.json yet. These thresholds are placeholders.")
        with st.expander("Action definitions"):
            for action, d in ACTION_DEFINITIONS.items():
                st.markdown("**%s**  \n%s %s" % (action.value, d["when"], d["boundary"]))

    choice = st.selectbox("Example", list(EXAMPLES))
    ex = EXAMPLES[choice]
    with st.form("scenario"):
        narrative = st.text_area("Scenario", value=ex["narrative"], height=160)
        c1, c2, c3, c4 = st.columns(4)
        stages = [s.value for s in SalesStage]
        stage = c1.selectbox("Sales stage", stages, index=stages.index(ex["sales_stage"]))
        sizes = ["small", "medium", "large"]
        size = c2.selectbox("Customer size", sizes, index=sizes.index(ex["customer_size"]))
        export = c3.checkbox("Export oriented", value=ex["export_oriented"])
        incumbent = c4.checkbox("Has incumbent supplier", value=ex["has_incumbent"])
        p1, p2 = st.columns(2)
        pains = p1.text_area("Pain points, one per line", value=ex["pain_points"], height=80)
        objs = p2.text_area("Objections, one per line", value=ex["objections"], height=80)
        submitted = st.form_submit_button("Recommend", type="primary")

    if not submitted:
        return
    if not narrative.strip():
        st.error("Enter a scenario.")
        return

    scenario = Scenario(id="app", narrative=narrative.strip(), sales_stage=stage,
                        customer_size=size, export_oriented=export, has_incumbent=incumbent,
                        pain_points=lines(pains), objections=lines(objs))
    names = ["rule", "llm"] + (["rag"] if retriever is not None else [])
    if retriever is None:
        st.warning("The knowledge base is empty, so the RAG system is not shown. "
                   "Add items to data/knowledge/.")

    ctx = Context(settings=settings, retriever=retriever,
                  run_id="app-%s" % datetime.date.today().strftime("%Y%m%d"),
                  raw_dir=paths.raw_dir(paths.RESULTS))

    for col, name in zip(st.columns(len(names)), names):
        with col:
            st.subheader(LABELS[name])
            try:
                with st.spinner("Running"):
                    out = SYSTEMS[name](scenario, ctx)
            except Exception as e:
                st.error("%s: %s" % (type(e).__name__, e))
                continue
            rec = out.recommendation
            if rec is None:
                st.error("No valid recommendation. %s" % (out.error or ""))
                continue
            st.markdown("### `%s`" % rec.action.value)
            st.write(rec.rationale)
            if name == "rule":
                st.caption("Deterministic rule on structured fields. No confidence gate.")
                continue
            decision, reason = gate(
                has_recommendation=True, confidence=rec.confidence,
                evidence_ids=rec.evidence_ids, retrieved_ids=out.retrieved_ids,
                top_score=out.top_score, tau_conf=tau_conf, tau_ret=tau_ret)
            st.metric("Confidence", "%.2f" % rec.confidence)
            if decision == PASS:
                st.success("Passes the gate")
            else:
                st.warning("Human review required: %s" % reason)
            if name == "rag":
                st.markdown("**Evidence cited**")
                if not rec.evidence_ids:
                    st.caption("None cited.")
                for eid in rec.evidence_ids:
                    item = by_id.get(eid)
                    with st.expander(eid if item else "%s (not in the retrieved set)" % eid):
                        if item:
                            st.write(item.text)
                            st.caption("type %s, source %s, version %s"
                                       % (item.type, item.source, item.version))
                with st.expander("Retrieved items (top score %.2f)" % (out.top_score or 0.0)):
                    for rid in out.retrieved_ids:
                        st.markdown("- `%s` %s" % (rid, by_id[rid].text[:160]))
            st.caption("tokens in %d, out %d, %.1f s" % (out.input_tokens, out.output_tokens, out.latency_s))


main()
