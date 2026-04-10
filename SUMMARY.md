# BI-FAST Unsupervised Fraud Detection — End-to-End Summary

Sistem deteksi fraud BI-FAST berbasis **unsupervised learning** (mimic Elastic DFA outlier detection), trained lokal lalu di-deploy ke Elasticsearch untuk inference real-time via ingest pipeline.

---

## 1. Arsitektur

```
                    ┌─────────────────────────────┐
                    │   Elasticsearch (8.14.1)    │
                    │   184.169.41.143:9200       │
                    └─────────────────────────────┘
                                  ▲
                                  │ fetch (scroll API)
                                  │
                ┌─────────────────┴─────────────────┐
                │  data/bifast_trx.csv (2006 rows)  │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────┐
        │  preprocessing.py                            │
        │  - engineer (log_amount, fee_ratio, geo,...) │
        │  - ColumnTransformer (Robust + OHE)          │
        └──────────────────┬───────────────────────────┘
                           ▼
        ┌──────────────────────────────────────────────┐
        │  train.py                                    │
        │  IsolationForest(n=200, contamination=0.05)  │
        │  → outlier_score [0,1], threshold P95        │
        └──────────────────┬───────────────────────────┘
                           ▼
        ┌──────────────────────────────────────────────┐
        │  upload_surrogate.py                         │
        │  Train RandomForestRegressor (mimic IForest) │
        │  Upload via eland → _ml/trained_models/...   │
        └──────────────────┬───────────────────────────┘
                           ▼
        ┌──────────────────────────────────────────────┐
        │  build_pipeline.py                           │
        │  Generate Painless preprocessing             │
        │  → _ingest/pipeline/fraud_unsupervised_...   │
        └──────────────────────────────────────────────┘
                           ▼
                   Inference di Elastic
        POST _ingest/pipeline/<name>/_simulate
```

---

## 2. Folder Structure

```
bank_ctbc/
├── data/
│   ├── bifast_trx.csv             # raw data dari index garda-context-bifast_trx
│   ├── bifast_trx_raw.json        # versi JSON
│   └── bifast_trx_scored.csv      # data + outlier_score + is_outlier
├── src/
│   ├── preprocessing.py           # feature engineering + ColumnTransformer
│   ├── train.py                   # train IsolationForest
│   ├── score.py                   # FraudScorer (local inference helper)
│   └── api.py                     # FastAPI server (opsional, mimic _infer)
├── elastic/
│   ├── upload_surrogate.py        # train RF surrogate + upload via eland
│   ├── build_pipeline.py          # generate ingest pipeline JSON
│   ├── put_pipeline.py            # PUT pipeline ke Elastic
│   ├── test_infer.py              # smoke test
│   ├── fraud_unsupervised_pipeline_v2.json
│   └── test_payloads.json         # kumpulan payload test (lihat #6)
├── models/
│   ├── fraud_iforest.joblib       # IsolationForest asli
│   ├── model_meta.json            # threshold, raw_min/max, dll
│   ├── surrogate_rf_v2.joblib     # surrogate yang di-upload
│   └── preprocess_params_v2.json  # params untuk Painless
├── fetch_data.py                  # script ambil data dari ES
└── SUMMARY.md                     # file ini
```

---

## 3. Workflow End-to-End

### Step 1 — Fetch data dari Elasticsearch

```bash
python fetch_data.py
```

Output: `data/bifast_trx.csv` (2006 dokumen, 32 kolom).

### Step 2 — Train model unsupervised lokal

```bash
python src/train.py
```

- Algorithm: **IsolationForest** (n_estimators=200, contamination=0.05)
- 49 fitur output (numerik + engineered + one-hot)
- Threshold: P95 → 0.7377
- Hasil: 102/2006 outlier (5%)

### Step 3 — Train surrogate model + upload ke Elastic

```bash
python elastic/upload_surrogate.py
```

- IsolationForest tidak natively didukung Elastic ML
- Surrogate: **RandomForestRegressor** (n=100, depth=8) yang meniru `outlier_score`
- Fit: MAE 0.0126, corr 0.9938
- Upload via `eland.ml.MLModel.import_model` → `_ml/trained_models/fraud_unsupervised_bifast_v2`

### Step 4 — Generate + PUT ingest pipeline

```bash
python elastic/build_pipeline.py
python elastic/put_pipeline.py
```

Pipeline berisi:
1. **Painless script** preprocessing (engineered features + RobustScaler + OneHot)
2. **Inference processor** → panggil model surrogate
3. **Painless threshold** → set `is_outlier`
4. **Cleanup** → hapus intermediate features

### Step 5 — Inference real-time

```
POST _ingest/pipeline/fraud_unsupervised_pipeline_v2/_simulate
```

---

## 4. Resources di Elastic

| Resource | Path |
|---|---|
| Model | `_ml/trained_models/fraud_unsupervised_bifast_v2` |
| Pipeline | `_ingest/pipeline/fraud_unsupervised_pipeline_v2` |
| Inference endpoint | `POST _ingest/pipeline/fraud_unsupervised_pipeline_v2/_simulate` |
| Auto-scoring index | `PUT <index>/_settings { "index.default_pipeline": "fraud_unsupervised_pipeline_v2" }` |

### Cek status

```
GET _ml/trained_models/fraud_unsupervised_bifast_v2
GET _ingest/pipeline/fraud_unsupervised_pipeline_v2
GET _ml/trained_models/fraud_unsupervised_bifast_v2/_stats
```

---

## 5. Output Format

```json
{
  "ml": {
    "outlier_score": 0.81,
    "is_outlier": true,
    "model_id": "fraud_unsupervised_bifast_v2"
  }
}
```

- `outlier_score` ∈ [0, 1] — semakin tinggi = semakin anomali
- `is_outlier` = `true` jika score ≥ **0.7377** (threshold P95)

---

## 6. Test Payloads

Lihat file [`elastic/test_payloads.json`](elastic/test_payloads.json) — berisi 12 use case yang siap dipakai di Kibana Dev Tools.

| # | Use Case | Expected |
|---|---|---|
| 1 | Normal small transaction | low score, not outlier |
| 2 | Normal medium retail | low score, not outlier |
| 3 | High amount intra-bank | medium score |
| 4 | Cross-bank besar tanpa fee | **high score, outlier** |
| 5 | Sangat besar (>1M USD eq) | **high score, outlier** |
| 6 | Suspicious device UNKNOWN | medium-high |
| 7 | Missing geo + missing device | medium-high |
| 8 | Channel tidak biasa (MB_JATIM) | medium |
| 9 | Foreign currency pattern | varies |
| 10 | Edge case amount=0 | varies |
| 11 | Real outlier dari training set | **high score** |
| 12 | Batch 5 doc untuk distribusi | mixed |

Cara pakai:
1. Buka file `elastic/test_payloads.json`
2. Copy isi `request_body` dari use case yang diinginkan
3. Paste ke Kibana Dev Tools setelah baris `POST _ingest/pipeline/fraud_unsupervised_pipeline_v2/_simulate`

---

## 7. Troubleshooting

| Masalah | Solusi |
|---|---|
| Latency call pertama 3000ms+ | Normal — Painless compile cost. Call ke-2 dst <300ms |
| `model_id not found` | Cek `GET _ml/trained_models/fraud_unsupervised_bifast_v2` |
| `pipeline not found` | Re-run `python elastic/put_pipeline.py` |
| Score selalu 0 | Field name di payload tidak match, cek `latitude/longitude` numeric |
| Mau retrain | `python src/train.py && python elastic/upload_surrogate.py && python elastic/build_pipeline.py && python elastic/put_pipeline.py` |

---

## 8. Versioning

| Version | Model | Pipeline | Trees × Depth | MAE |
|---|---|---|---|---|
| v1 | `fraud_unsupervised_bifast` | `fraud_unsupervised_pipeline` | 200 × 12 | 0.0032 |
| **v2** (active) | `fraud_unsupervised_bifast_v2` | `fraud_unsupervised_pipeline_v2` | 100 × 8 | 0.0126 |

Untuk redeploy versi baru, set `MODEL_VERSION` env var:

```bash
MODEL_VERSION=v3 python elastic/upload_surrogate.py
MODEL_VERSION=v3 python elastic/build_pipeline.py
MODEL_VERSION=v3 python elastic/put_pipeline.py
```
