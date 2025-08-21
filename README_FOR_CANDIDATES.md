## **Mini-Prod ML Challenge (3–4 hours)**

**Goal:** Ship a small but production-minded binary classifier on a provided tabular dataset, with training, serving, monitoring, and a light “agentic” check. Keep it simple; correctness \> fanciness.

### **What we provide**

* A starter repo skeleton.  
* Data files: `adult_synth.csv` (main dataset), `adult_ref_sample.csv`, `adult_shifted_sample.csv`, `metrics_history.jsonl`, `drift_latest.json`.

### **Your deliverables**

* Reproducible training script that saves model \+ metrics.  
* FastAPI service with `/health` and `/predict`.  
* Minimal drift check CLI.  
* Agentic Monitor CLI (LLM-optional).  
* Basic tests, Dockerfile, CI skeleton, and a 1-page GCP design note.  
* A 15 min video showing your work.

---

## **Tasks**

### **Part A — Train (core ML)**

Implement `python -m src.train --data data/adult_synth.csv --outdir artifacts/` that:

* Splits train/val with fixed seed.  
* Preprocesses (impute, encode categoricals, scale as needed).  
* Trains one strong model: XGBoost/LightGBM **or** a well-tuned logistic regression.  
* Logs metrics (ROC-AUC, PR-AUC, Accuracy) on **val** into `artifacts/metrics.json`.  
* Saves `model.pkl`, `feature_pipeline.pkl`, `feature_importances.csv` (or SHAP summary).  
* (Optional) 10–20 trial randomized HPO.

**Acceptance:** ROC-AUC ≥ **0.83** on our split; artifacts saved.

#### Addendum:

***Fairness & Sensitive Features (required)***

* Treat **`race`, `sex`, `native_country`, `marital_status`** as **sensitive**.  
* **Do not use** them as model inputs. Keep them only to compute fairness metrics.  
* After you train, report:  
  * Overall ROC-AUC and by group for **sex** and **race**.  
  * At decision threshold 0.5: TPR and FPR by group.  
  * One of these two disparity metrics:   
    * Demographic Parity Difference (|P(ŷ=1|group)−P(ŷ=1|ref)|)   
    * Equal Opportunity Difference (|TPR\_group−TPR\_ref|).  
* If any gap \> 0.10, add one sentence on mitigation you’d try (drop/transform features, threshold per group, reweighting, post-processing). No need to implement.

---

### **Part B — Serve (FastAPI)**

`uvicorn src.app:app` with:

* `GET /health` → `{"status":"ok"}`  
* `POST /predict` → accepts a JSON list of rows (Pydantic schema), returns probabilities \+ class.  
* Clear 400s for missing/invalid fields or unknown categories.

---

### **Part C — MLOps bits**

* **Logging:** `artifacts/metrics.json` with metrics, timestamp, and git SHA (if available).  
* **Tests (pytest):**  
  * `tests/test_training.py` sanity-checks artifacts exist and ROC-AUC ≥ 0.83.  
  * `tests/test_inference.py` boots the API locally and checks `POST /predict` on 2 sample rows returns probs in \[0,1\].  
* **Docker:** image that can train and serve.  
* **CI (GitHub Actions):** install, run tests, build Docker on push.

---

### **Part D — Monitoring (drift mini-check)**

`python -m src.drift --ref data/adult_ref_sample.csv --new data/adult_shifted_sample.csv`  
Outputs `artifacts/drift_report.json` with PSI/KS per feature and:

```json
{"threshold": 0.2, "overall_drift": true|false, "features": {"age": 0.23, ...}}
```

**Part E — Agentic Monitor (LLM-optional)**

Implement a tiny “monitoring agent” that **observes → thinks → acts**:

**CLI:**

```shell
python -m src.agent_monitor \
  --metrics data/metrics_history.jsonl \
  --drift data/drift_latest.json \
  --out artifacts/agent_plan.yaml
```

**Behavior:**

* **Observe:** load metrics history \+ latest drift report.  
* **Think (rules or LLM-optional):** classify status: `healthy|warn|critical`. Suggested heuristics:  
  * `warn` if ROC-AUC drops ≥ 3% vs 7-day median **or** p95 latency \> 400ms for 2 consecutive points.  
  * `critical` if drop ≥ 6% **or** (`overall_drift` true **and** PR-AUC down ≥ 5%).  
* **Act:** emit an **action plan** (YAML) with `status`, `findings`, `actions` (subset of):  
  * `open_incident`, `trigger_retraining`, `roll_back_model`, `raise_thresholds`, `page_oncall=false`, `do_nothing`.  
* **Optional:** `POST /monitor` returns the same payload.

Example output:

```
status: warn
findings:
  - roc_auc_drop_pct: 3.8
  - latency_p95_ms: 412
  - drift_overall: false
actions:
  - trigger_retraining
  - raise_thresholds
rationale: >
  AUC fell 3.8% vs 7-day median; p95 latency > 400ms for two windows.
```

## **Deliverables checklist**

* `src/` code, `tests/`, `docker/Dockerfile`, `.github/workflows/ci.yml`, `requirements.txt`.  
* `artifacts/metrics.json` after training; `artifacts/drift_report.json`; `artifacts/agent_plan.yaml`.  
* `design_gcp.md` (≤1 page): how you’d run on GCP (BigQuery, training on Vertex AI or GKE, serving on Cloud Run/Vertex, metrics in Cloud Monitoring/Grafana), brief cost notes.  
* `README.md` with quickstart: `make train`, `make serve`, `make test`.

## **Constraints**

* No internet required for training/eval.  
* Keep secrets out of code/CI.  
* Deterministic seeds.

## **Submission**

* GitHub repo link \+ brief README.  
* Commands to reproduce:  
  * `python -m src.train --data data/adult_synth.csv --outdir artifacts/`  
  * `uvicorn src.app:app --port 8000`  
  * `python -m src.drift --ref data/adult_ref_sample.csv --new data/adult_shifted_sample.csv`  
  * `python -m src.agent_monitor --metrics data/metrics_history.jsonl --drift data/drift_latest.json --out artifacts/agent_plan.yaml`  
  * `pytest -q`