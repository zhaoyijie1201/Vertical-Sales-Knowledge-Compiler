# Evaluations

Every number below can be recomputed without an API key: `python scripts/report.py` reads
the logged runs in `results/` and writes the tables to `results/tables/`.

## 1. What is compared

| System | Sees |
|---|---|
| Majority class | Nothing; always answers the most common gold label |
| Rule baseline | Structured fields only, 15 ordered rules |
| Generic LLM | The scenario, with an empty knowledge block |
| Vertical RAG + LLM | The scenario and the 5 retrieved knowledge items |

The generic LLM and the RAG system use the same prompt; the knowledge block is the only
difference (a test enforces this).

## 2. The evals in this repository

| Eval | Question it answers | Script | Output |
|---|---|---|---|
| Next-best-action accuracy | How often is the recommended action the gold action? | `scripts/run_eval.py`, `scripts/report.py` | T1 |
| Paired comparison | Is the difference between two systems more than chance? Exact McNemar test on the same scenarios | `scripts/report.py` | T6 |
| Accuracy by subset | Where does the knowledge base help? By supplier-fact dependence, label certainty, stage, size | `scripts/report.py` | T3 |
| Abstention | How often does the gate escalate, and are escalated cases the ones that would be wrong? | `scripts/tune_gate.py`, `scripts/report.py` | T2 |
| Leakage | Does the input already contain the answer? Score before and after removing leaking sentences | `scripts/leakage_check.py` | summary |
| Retrieval recall | Is the needed knowledge item among the 5 retrieved? | `scripts/eval_retrieval.py` | table |
| Label agreement | Do two independent annotators agree on the held-out labels? | `scripts/second_annotator.py` | Cohen's kappa |
| Cost and latency | What does one recommendation cost? | `scripts/report.py` | T4 |
| Configuration time | How long does an FDE take to write an agent instruction, by hand and with the system? | stopwatch; `results/study/` | README |

## 3. Discipline

- Labels before knowledge: dev labels committed before the knowledge base (`a4dde59`, then `5f75df5`).
- Held-out labels committed before any system ran on them (`a694276`).
- Held-out is run once. `run_eval.py` refuses a second run unless `--force --reason` is given; the reason is recorded.
- Gate thresholds must exist before held-out runs; `tune_gate.py` refuses a held-out run.
- A reply that fails validation twice counts as wrong.
- Every model call, including failed attempts, is logged in `results/raw/`.

## 4. Results

### Dev, 40 scenarios (tuning only)

| Majority | Rule | Generic LLM | RAG |
|---|---|---|---|
| 12.5% | 45.0% | 95.0% | 100% |

### Held-out, 80 scenarios (run `heldout-raw-20261002`)

| System | Accuracy | 95% CI |
|---|---|---|
| Majority class | 28.7% | 20.0 to 39.5 |
| Rule baseline | 36.2% | 26.6 to 47.2 |
| Generic LLM | 72.5% | 61.9 to 81.1 |
| Vertical RAG + LLM | 78.8% | 68.6 to 86.3 |

| Subset | Rule | Generic LLM | RAG | RAG vs LLM, exact McNemar p |
|---|---|---|---|---|
| All, 80 | 36.2% | 72.5% | 78.8% | 0.302 |
| Depends on a supplier fact, 32 | 34.4% | 56.2% | 71.9% | 0.267 |
| No supplier fact, 48 | 37.5% | 83.3% | 83.3% | 1.000 |
| Clear label, 30 | 30.0% | 66.7% | 93.3% | **0.008** |
| Judgment label, 50 | 40.0% | 76.0% | 70.0% | 0.453 |

Both model systems beat the rule baseline (p < 0.001).

### Gate on held-out

| Abstain rate | Escalated cases that would be wrong | Accuracy on answered cases |
|---|---|---|
| 81.3% (65/80) | 26.2% (17/65) | 100% (15/15) |

### Retrieval recall on dev

| k = 3 | k = 5 | k = 8 |
|---|---|---|
| single query 11/15, fused 14/15 | single 12/15, fused 15/15 | single 14/15, fused 15/15 |

## 5. Critique of the evals

- **Small samples.** 80 held-out scenarios give 95% intervals about 18 points wide. The
  overall RAG advantage of 6 points is inside the noise; only the clear-label subset shows a
  significant difference.
- **Run-to-run variation.** The leakage-stripped run changed only 3 sentences, none of which
  changed correctness, yet RAG scored 83.8% and the generic LLM 68.8%. All changes happened
  on unchanged scenarios: the model does not answer identically twice. A few points between
  systems are within this variation. The first run is the reported result.
- **Label quality.** 50 of 80 held-out labels are judgment calls. On these, the systems and
  the second annotator disagree more, and RAG does not beat the generic LLM. The clear-label
  result carries more weight.
- **Independent data sources.** Held-out was generated and second-annotated by models from
  two vendors other than that of the model under test, so the test scenarios and the check
  on their labels do not share the tested model's blind spots.
- **The gate did not transfer.** Thresholds came from the 10th percentile of dev confidence
  and retrieval score. Dev narratives are clean and close to the knowledge base wording, so
  scores were high; held-out narratives are informal notes, so 81% of cases fell below. The
  escalated cases were only slightly more likely to be wrong (26% vs 21% overall). Thresholds
  were not re-tuned on held-out. A gate needs calibration data that looks like production.
- **Synthetic data.** No result here was checked against real accounts. The results show
  how the systems behave on this data and that knowledge helps where a supplier fact decides
  the answer; they do not show performance on live customers.
- **Self-timed study.** The configuration-time study has six scenarios, timed and judged by
  the developer.

## 6. Tuning done on dev

| Change | Evidence | Run |
|---|---|---|
| Single query to fused retrieval | Recall at k = 5 from 12/15 to 15/15 | `scripts/eval_retrieval.py` |
| Revised dev narratives | Generic LLM fell from 100% to 95%; narratives no longer gave the answer away | archive vs current gold |
| Ninth action `proceed_to_order` | About ten held-out scenarios had no open barrier; added before the held-out run, dev re-run unchanged | `dev-raw-20260930-c` |
| Gate rule: percentile instead of accuracy | RAG made no error on dev, so accuracy-based thresholds could not be learned | `results/gate.json` |
