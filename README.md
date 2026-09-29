# Vertical Sales Knowledge Compiler

Turns vertical sales knowledge into a grounded next-best-action recommendation for a
Forward-Deployed Engineer (FDE) who configures a B2B sales agent. Domain: industrial
sensors, export-oriented manufacturing. Course project for PE6201, NTU.

```
one customer scenario -> one retrieval step -> one model call -> one structured recommendation
```

The output is an action from a closed set of eight, a short rationale, the knowledge
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

4. Demo interface.

   ```bash
   streamlit run src/vskc/app.py
   ```

Add `--mock` to `run_eval.py` to exercise the pipeline with deterministic fake replies.
Mock runs are labeled as such and excluded from the summary.

## Layout

```
src/vskc/        schema, retriever, prompts, model gateway, three systems, gate, metrics
scripts/         data validation, leakage check, evaluation, threshold tuning, report, generation
data/gold/       40 labeled scenarios, fixed before the knowledge base
data/knowledge/  playbook, product records, customer SOPs, past outcomes
data/scenarios/  dev and held-out scenario files, raw and leakage-stripped
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
- **Generation never labels.** The scenario generator has no target action. A person
  labels the generated scenarios.
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

## Build or rent

| Layer | Choice | Own or rent |
|---|---|---|
| Interface | Streamlit | own |
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
- Gold labels reflect one person's judgement. There is no second annotator.
- The silent failure to watch for is a recommendation that reads as commercially
  plausible and is wrong for the customer's industry. The gate escalates low-confidence
  and uncovered cases, and the report states how many escalated cases would have been
  wrong.
- Knowledge text is treated as data. The prompt tells the model not to follow
  instructions inside it, and the output is restricted to the closed action set.
