# Vertical Sales Knowledge Compiler

Turns vertical sales knowledge into a grounded next-best-action recommendation for a
Forward-Deployed Engineer (FDE) who configures a B2B sales agent. Domain: industrial
sensors, export-oriented manufacturing. Course project for PE6201, NTU.

```
one customer scenario -> one retrieval step -> one model call -> one structured recommendation
```

The output is an action from a closed set of nine, a short rationale, the knowledge
items it relied on, a confidence score, and a human-review flag.

## What is compared

| System | What it sees |
|---|---|
| Rule-based baseline | Structured fields only. No model. |
| Generic LLM | The scenario. Same prompt as below, with an empty knowledge block. |
| Vertical RAG + LLM | The scenario plus the top-k knowledge items from BM25 retrieval. |

Accuracy is exact match against the gold label, reported next to the
majority-class baseline.

## Reproduce the tables without an API key

Every model call is logged under `results/raw/` and every prediction under
`results/runs/`. The report is built from those files only.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows.  macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python scripts/report.py
```

Tables are written to `results/tables/`.

## Run the tests

```bash
pytest
```

The tests use mock replies and small fixture files. They call no model and need no key.

## Run it yourself

1. Copy `.env.example` to `.env`. Set the API key and the three model slugs.
2. Check the setup. This makes no model call.

   ```bash
   python scripts/check_setup.py
   python scripts/check_setup.py --models anthropic
   ```

3. Validate the data, evaluate on dev, choose thresholds, then run held-out once.

   ```bash
   python scripts/validate_data.py
   python scripts/eval_retrieval.py
   python scripts/leakage_check.py --split dev
   python scripts/run_eval.py --split dev
   python scripts/run_eval.py --split dev --variant stripped
   python scripts/tune_gate.py --run-id <dev run id>

   python scripts/leakage_check.py --split heldout
   python scripts/run_eval.py --split heldout
   python scripts/run_eval.py --split heldout --variant stripped
   python scripts/report.py
   ```

4. Web interface.

   ```bash
   python -m vskc.api
   ```

   Open http://127.0.0.1:8000. The **Workbench** replays what each system answered in the
   evaluation run (no model call, no cost), or calls the model live on a held-out, dev or
   custom scenario. Each answer shows the action, the rationale, the confidence, the gate
   decision and the cited knowledge items. **Results** shows the held-out accuracy,
   subsets, paired tests, the gate and cost. The page is served by FastAPI and is plain
   HTML and JavaScript with no build step; the JSON API is documented at `/api/docs`.
   Deep links such as `/#heldout/ho-022`, `/#results` and `/#heldout/ho-022/hide` (gold
   labels hidden) are useful for demonstrations.

   A simpler Streamlit page is kept as a fallback: `streamlit run src/vskc/app.py`.

Add `--mock` to `run_eval.py` to exercise the pipeline with deterministic fake replies.
Mock runs are labeled as such and excluded from the summary.

## Layout

```
src/vskc/        schema, retriever, prompts, model gateway, three systems, gate, metrics
src/vskc/api.py  FastAPI app: JSON API and the web page in src/vskc/web/
scripts/         data validation, leakage check, evaluation, threshold tuning, report, generation
data/gold/       40 labeled scenarios, fixed before the knowledge base
data/knowledge/  playbook, product records, customer SOPs, past outcomes
data/scenarios/  dev and held-out scenario files, raw and leakage-stripped
results/annotation/  second annotator's labels
results/study/   configuration-time study: timings and the model calls made
data/study/      the six configuration-time study scenarios
results/raw/     one line per model call, including failed attempts
results/runs/    predictions and run metadata
results/tables/  generated tables
tests/           unit tests and an end-to-end mock run
```

## Evaluation discipline

- **Labels before knowledge.** The 40 gold labels were fixed and committed before any
  knowledge item existed. See the commit history.
- **Held-out is run once.** `run_eval.py` refuses a second held-out run unless `--force`
  is given with a reason, which is recorded in the run metadata.
- **Thresholds are fixed first.** Held-out will not run until `results/gate.json` exists.
  `tune_gate.py` refuses to tune on a held-out run.
- **Leakage is measured.** Sentences that name an action or give advice are removed to
  produce a stripped variant. Scores are reported before and after.
- **Failures stay in the denominator.** A reply that fails validation twice counts as a
  wrong answer.
- **Generation never labels.** The scenario generator has no target action. Labels are
  assigned afterwards, and every label is reviewed by a person before any system is run
  on the file.
- **Held-out labels were checked.** A second annotator, a model from a different vendor
  than the model under test, labeled all 80 held-out scenarios independently with the
  whole knowledge base in view. Agreement is 78.8 percent, Cohen's kappa 0.74. Each label
  is marked `clear` or `judgment`, and accuracy is reported for both groups. Reproduce
  with `python scripts/second_annotator.py --split heldout`.
- **Same prompt for both model systems.** The generic LLM and the RAG system differ only
  in whether the knowledge block is empty.

## Design decisions fixed before the held-out set existed

These were committed before any held-out scenario was generated.

**Held-out composition.** 80 scenarios. At least 40 percent carry a customer requirement
whose consequence depends on a fact that only the knowledge base states, such as a
product's temperature limit or the minimum quantity for a custom housing. Each such
requirement is generated in two versions, one the supplier can meet and one it cannot,
so the fact changes the right action. The generator is told the customer's requirement
only. It is never told what the supplier can do. Accuracy is reported separately for
scenarios that depend on a supplier fact and for those that do not.

Reason: on the 40 dev scenarios a model with no knowledge base already scores 95
percent, and every one of its errors is a scenario that needs a supplier fact. A
held-out set without such scenarios could not show whether the knowledge base matters.

**Retrieval.** BM25 with two queries per scenario, the narrative and the pain point and
objection phrases, merged by reciprocal rank fusion, top 5. On the 15 dev scenarios that
need a supplier fact this retrieves the needed item for 15, against 12 for a single
query. Reproduce with `python scripts/eval_retrieval.py`. The mode was chosen on dev,
so that recall is optimistic.

**Gate thresholds.** The RAG system made no wrong answer on dev, so thresholds cannot be
learned from errors. They are set to the 10th percentile of the dev confidence and of
the dev retrieval score: a case is escalated when the system is less confident, or the
knowledge base matches less well, than on nine in ten dev cases. Values are in
`results/gate.json`.

**Ninth action.** `proceed_to_order` was added on 2026-09-30, after the held-out
scenarios were generated and before any system was run on them. The first eight actions
each address a barrier. Reading the generated scenarios showed that about one in eight has
no barrier: requirements met, terms accepted, customer ready to order. Without a ninth
action those scenarios would have no correct label. The 40 dev labels are unchanged, and
dev was re-run with the nine-action prompt (prompt version p2) before thresholds were set.

## FDE configuration-time study

The problem statement claims the system reduces the effort of configuring a sales agent.
Accuracy does not measure that, so I timed it.

**Design.** Six new scenarios, none in dev or held-out, generated by
`scripts/gen_config_study.py` into `data/study/config_study.jsonl`. Two groups of three,
matched by type: one decision that depends on a supplier fact, one price objection, one
other concern. For each scenario I wrote a sales-agent instruction meeting the same
seven-point standard: next action, rationale, knowledge cited, what to tell the customer,
what not to promise, whether to escalate, and a self-check against the scenario. Group A
by hand with only the knowledge base text; group B with the web interface in live mode.
The two conditions alternated (manual, system, manual, ...) so that growing familiarity
was shared between them. Time ran on a stopwatch from the first word read to the last
checklist item met.

| Pair | By hand | With the system | Saved |
|---|---|---|---|
| Supplier fact (cs-01, cs-02) | 10.70 min | 5.57 min | 5.13 min |
| Price objection (cs-03, cs-04) | 11.50 min | 4.43 min | 7.07 min |
| Other concern (cs-05, cs-06) | 9.75 min | 4.57 min | 5.18 min |
| **Mean** | **10.65 min** | **4.86 min** | **5.79 min, 54%** |

All six instructions met the standard, and I adopted the system's recommendation in all
three system cases. The model's response took 2.5 to 3.8 seconds, about 1 percent of the
system condition; the rest was reading, entering the scenario, checking the cited
evidence and writing the instruction. The RAG call cost about USD 0.016 per scenario.

Records: `results/study/config_time_study.csv`; the model calls made in the system
condition are in `results/study/system_condition_calls_20261002.jsonl`.

Limits: six scenarios, so this shows a magnitude, not a tested difference. I am the
developer, I know the knowledge base and the action set, and I judged the quality of my
own instructions. The two groups used different scenarios, matched by type but not
identical in difficulty. Two generated scenarios had one product detail corrected to
match the knowledge base; the edits are recorded in `meta.edits`.

## Build or rent

| Layer | Choice | Own or rent |
|---|---|---|
| Interface | Single HTML page, no framework | own |
| Serving | FastAPI JSON API, so the recommendation can be called by an execution-layer platform | own (FastAPI and Uvicorn are rented libraries) |
| Orchestration | Python | own |
| Knowledge schema | Pydantic models, closed action set | own |
| Retrieval | BM25, implemented in `retriever.py` | own |
| Strategy compiler | Prompt plus JSON Schema output | own |
| Confidence gate | Deterministic rules | own |
| Evaluation | Scripts in this repository | own |
| Foundation model | Hosted model through OpenRouter, slug in `.env` | rent |
| Prospecting, CRM, outreach | Existing sales platforms | rent, out of scope |

## Intended use and limits

Intended use: decision support for FDEs and sales implementation teams.

Not for: contacting customers, negotiating prices, making binding commercial
commitments, or replacing human approval of strategic decisions. The system has no tool
that acts on any external system.

Known limits:

- All data is synthetic. No real customer data is used. Results say how the systems
  behave on this data, not how they would behave in a live account.
- Labels reflect one reviewer's judgement. On held-out, 50 of the 80 labels are judgement
  calls where another action is defensible; the recorded alternative is in each record's
  `meta`. Results on the 30 clear labels carry more weight.
- Leakage was negligible: the filter removed 3 sentences from 3 held-out scenarios, and
  neither model system changed its correctness on those three. The stripped run therefore
  works as a second run on nearly unchanged input. Between the two runs the generic LLM
  changed its answer on 5 of 80 scenarios and the RAG system on 4. Every change in
  correctness, 3 for the generic LLM and 4 for RAG, happened on unchanged scenarios.
  Differences of a few points between systems are within this run-to-run variation. The
  first held-out run is the reported result; the second is reported alongside it.
- Held-out scenarios are generated from independently sampled attributes, so some
  combinations are unusual, and in five of them the generator stated a supplier-side
  fact that conflicts with the knowledge base. They were kept as generated and flagged.
- The silent failure to watch for is a recommendation that reads as commercially
  plausible and is wrong for the customer's industry. The gate escalates low-confidence
  and uncovered cases, and the report states how many escalated cases would have been
  wrong.
- Knowledge text is treated as data. The prompt tells the model not to follow
  instructions inside it, and the output is restricted to the closed action set.
