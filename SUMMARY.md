# BI-FAST Fraud Detection V3 — End-to-End Summary

Sistem deteksi fraud BI-FAST berbasis **unsupervised learning** (IsolationForest) yang di-deploy **fully ES-native** di Elasticsearch. Inference dilakukan via ingest pipeline dengan velocity enrichment real-time, dan dipanggil dari Postman collection [`postman/fraud_detection_v3.json`](postman/fraud_detection_v3.json).

---

## 1. Arsitektur End-to-End

```
                        ┌──────────────────────────────┐
                        │   Elasticsearch 8.14.1       │
                        │   184.169.41.143:9200        │
                        └──────────────────────────────┘
                                       ▲
   ┌───────────────────────────────────┼───────────────────────────────────┐
   │ TRAINING (offline, lokal)         │ INFERENCE (online, ES-native)     │
   │                                   │                                   │
   │  garda-context-bifast_trx         │  Postman fraud_detection_v3.json  │
   │           │                       │           │                       │
   │           ▼                       │           ▼                       │
   │  fetch_data.py                    │  POST _ingest/pipeline/           │
   │           │                       │   fraud_unsupervised_pipeline_    │
   │           ▼                       │   v3_enriched/_simulate           │
   │  data/bifast_trx.csv              │           │                       │
   │           │                       │           ▼                       │
   │           ▼                       │  ┌─────────────────────────────┐  │
   │  src/preprocessing.py             │  │ 1. enrich processor         │  │
   │  (engineered features +           │  │    lookup acct-velocity-    │  │
   │   ColumnTransformer)              │  │    snapshot by sourceAccount│  │
   │           │                       │  │ 2. painless preprocessing   │  │
   │           ▼                       │  │    (engineered + scaling +  │  │
   │  src/train_v3.py                  │  │     one-hot)                │  │
   │  IsolationForest(n=200,           │  │ 3. inference processor →    │  │
   │  contamination=0.05)              │  │    fraud_unsupervised_      │  │
   │  + threshold calibration          │  │    bifast_v3 (RF surrogate) │  │
   │           │                       │  │ 4. band assignment          │  │
   │           ▼                       │  │    (LOW / MEDIUM / HIGH)    │  │
   │  models/fraud_iforest_v3.joblib   │  └─────────────────────────────┘  │
   │  models/model_meta_v3.json        │           │                       │
   │           │                       │           ▼                       │
   │           ▼                       │   { ml.outlier_score,             │
   │  elastic/upload_surrogate.py      │     ml.is_anomaly,                │
   │  RandomForestRegressor surrogate  │     ml.band }                     │
   │  + eland.MLModel.import_model     │                                   │
   │           │                       │   ▲                               │
   │           ▼                       │   │ continuous transform          │
   │  _ml/trained_models/              │   │ acct_velocity_agg             │
   │  fraud_unsupervised_bifast_v3     │   │ (refresh 1m)                  │
   │           │                       │   │                               │
   │           ▼                       │   acct-velocity-snapshot          │
   │  elastic/setup_v3_enriched.py     │   (sourceAccount → amt_sum_2m,    │
   │  (provision transform + enrich    │    5m, 1h, 1d, tx_count_5m, 1h,   │
   │   policy + pipeline)              │    distinct_device/ip/location)   │
   └───────────────────────────────────┴───────────────────────────────────┘
```

---

## 2. Folder Structure

```
bank_ctbc/
├── data/
│   ├── bifast_trx.csv                   # raw dari index garda-context-bifast_trx
│   ├── bifast_trx_raw.json
│   └── bifast_trx_scored.csv            # data + outlier_score + band
├── src/
│   ├── preprocessing.py                 # feature engineering + ColumnTransformer
│   ├── train_v3.py                      # train IForest + kalibrasi threshold
│   └── archive/                         # versi lama (v1, v2)
├── elastic/
│   ├── transform_acct_velocity.json     # continuous transform definition
│   ├── enrich_policy_acct_velocity.json # enrich policy (match by sourceAccount)
│   ├── fraud_unsupervised_pipeline_v3_enriched.json  # ingest pipeline aktif
│   ├── upload_surrogate.py              # train RF surrogate + upload via eland
│   ├── setup_v3_enriched.py             # orchestrator end-to-end provisioning
│   └── archive/                         # pipeline v1/v2, build/put scripts lama
├── models/
│   ├── fraud_iforest_v3.joblib          # IsolationForest asli
│   ├── surrogate_rf_v3.joblib           # surrogate RF (yang di-upload)
│   ├── preprocess_params_v3.json        # params Painless (scaler, kategori OHE)
│   └── model_meta_v3.json               # threshold, band, kalibrasi
├── postman/
│   ├── fraud_detection_v3.json          # ★ Postman collection AKTIF
│   └── archive/                         # koleksi sebelumnya
├── fetch_data.py
└── SUMMARY.md
```

---

## 3. Komponen di Elasticsearch

| Resource | ID / Path | Fungsi |
|---|---|---|
| Source index | `garda-context-bifast_trx` | Sumber transaksi BI-FAST mentah |
| Continuous transform | `_transform/acct_velocity_agg` | Agregasi velocity per `sourceAccount`, refresh tiap 1 menit |
| Snapshot index | `acct-velocity-snapshot` | Output transform — di-lookup oleh enrich |
| Enrich policy | `_enrich/policy/acct_velocity_policy` | Match `sourceAccount` → inject velocity fields |
| Trained model | `_ml/trained_models/fraud_unsupervised_bifast_v3` | RandomForest surrogate dari IsolationForest |
| Ingest pipeline | `_ingest/pipeline/fraud_unsupervised_pipeline_v3_enriched` | Endpoint inference (4-step processor chain) |

### Velocity fields dari enrichment

`amt_sum_2min`, `amt_sum_5min`, `amt_sum_1h`, `amt_sum_1d`, `tx_count_5min`, `tx_count_1h`, `distinct_device_5min`, `distinct_ip_5min`, `distinct_location_5min`.

### Pipeline `fraud_unsupervised_pipeline_v3_enriched`

1. **enrich processor** — lookup `acct-velocity-snapshot` by `sourceAccount`, merge velocity fields ke dokumen.
2. **painless preprocessing** — bangun engineered features (`log_amount`, `fee_ratio`, `cross_bank`, geo, dst), apply `RobustScaler`, one-hot encode kategori — semuanya dari `models/preprocess_params_v3.json`.
3. **inference processor** — panggil `fraud_unsupervised_bifast_v3`, hasil di `ml.predicted_value`.
4. **band assignment painless** — set `ml.outlier_score`, `ml.is_anomaly`, dan `ml.band` (LOW / MEDIUM / HIGH) berdasarkan threshold.

---

## 4. Workflow Provisioning

```bash
# (1) Fetch data dari ES
python fetch_data.py

# (2) Train IsolationForest + kalibrasi threshold (lokal)
python src/train_v3.py

# (3) Train surrogate RF + upload model ke ES (via eland)
MODEL_VERSION=v3 python elastic/upload_surrogate.py

# (4) Provision transform + enrich policy + pipeline + smoke test
python elastic/setup_v3_enriched.py            # full setup
python elastic/setup_v3_enriched.py --upload   # full setup + re-upload model
python elastic/setup_v3_enriched.py --smoke    # smoke test only
```

`setup_v3_enriched.py` adalah orchestrator idempotent — stop/delete resource lama lalu PUT ulang transform → enrich policy → pipeline, dan diakhiri smoke test `_simulate`.

---

## 5. Model & Threshold (v3)

Dari [`models/model_meta_v3.json`](models/model_meta_v3.json):

| Field | Value |
|---|---|
| Algorithm | IsolationForest, n_estimators=200, contamination=0.05 |
| Score normalization | `neg_score` min-max → [0, 1] (raw_min −0.6211, raw_max −0.3304) |
| Threshold MEDIUM | **0.6184259705500809** |
| Threshold HIGH | **0.7309104134367547** |
| Kalibrasi | `f1_max_vs_garda_rules` (F1=0.594, P=0.423, R=1.0) terhadap label rule-based Garda |
| n_train | 2006 |
| n_outliers MEDIUM | 201 (~10%) |
| n_outliers HIGH | 41 (~2%) |

Bands:
- **HIGH** (`score ≥ 0.7309`) → auto-block
- **MEDIUM** (`0.6184 ≤ score < 0.7309`) → review / step-up auth
- **LOW** (`score < 0.6184`) → pass

Rule-based label Garda yang dipakai untuk kalibrasi: `amt_sum_2min ≥ 350M`, `amt_sum_5min ≥ 100M`, `amt_sum_1d ≥ 150M`, `tx_count_5min ≥ 3`, `tx_count_1h ≥ 51`, `distinct_*_5min ≥ 2`.

---

## 6. Postman Collection — `postman/fraud_detection_v3.json`

Collection aktif untuk testing & demo end-to-end.

### Variables

| Key | Default |
|---|---|
| `esHost` | `https://184.169.41.143:9200` |
| `esUser` | `elastic` |
| `esPass` | `p@ssw0rd` |
| `pipeline` | `fraud_unsupervised_pipeline_v3_enriched` |

Auth: HTTP Basic (variabel di atas).

### Pre-request script

Setiap request otomatis membungkus tiap dokumen di `docs[]` dengan `{ "_source": { ... } }` jika belum ada — jadi kamu cukup menulis payload **flat 23 field** tanpa khawatir format `_simulate`.

### Requests

| # | Name | Method | Endpoint | Tujuan |
|---|---|---|---|---|
| 1 | **Score - LOW** | POST | `{{esHost}}/_ingest/pipeline/{{pipeline}}/_simulate` | Transaksi normal kecil (IDR 15.500, KOMIMOBILE) → expected `band: LOW` |
| 2 | **Score - MEDIUM** | POST | `…/_simulate` | RECEIVING IDR 75.000.000 cross-bank → expected `band: MEDIUM` |
| 3 | **Score - HIGH** | POST | `…/_simulate` | Sama seperti #2 tapi `destinationAccountType: CV` (corporate) → expected `band: HIGH` |
| 4 | **Score - Batch** | POST | `…/_simulate` | 3 dokumen sekaligus untuk uji distribusi LOW/MEDIUM/HIGH |
| 5 | Pipeline definition | GET | `{{esHost}}/_ingest/pipeline/{{pipeline}}` | Inspect pipeline aktif |
| 6 | Transform stats | GET | `{{esHost}}/_transform/acct_velocity_agg/_stats` | Cek health continuous transform |
| 7 | Enrich policy | GET | `{{esHost}}/_enrich/policy/acct_velocity_policy` | Verifikasi enrich policy |
| 8 | Trained model stats | GET | `{{esHost}}/_ml/trained_models/fraud_unsupervised_bifast_v3/_stats` | Cek model deploy & inference count |
| 9 | Velocity snapshot sample | GET | `{{esHost}}/acct-velocity-snapshot/_search?size=10` | Inspect data velocity per akun |

### Payload format (23 field flat)

```json
{
  "docs": [
    {
      "bifastId": "...",
      "transactionId": "...",
      "transactionDirection": "ORIGIN" | "RECEIVING",
      "channel": "KOMIMOBILE" | "BIFAST" | ...,
      "transactionType": "CT",
      "sourceBic": "...", "sourceAccount": "...", "sourceAccountType": "SVGS",
      "sourceCountryCode": "",
      "destinationBic": "...", "destinationAccount": "...", "destinationAccountType": "SVGS",
      "destinationCountryCode": "",
      "currency": "IDR", "amount": 75000000, "fee": 0,
      "chargeType": "D", "chargeBearer": "DEBT", "countryCode": "ID",
      "deviceId": "SAMSUNG", "ipAddress": "10.10.20.15",
      "latitude": -6.1717714, "longitude": 106.7913427
    }
  ]
}
```

### Response shape

```json
{
  "docs": [{
    "doc": {
      "_source": {
        "...original + velocity fields...",
        "ml": {
          "outlier_score": 0.81,
          "is_anomaly": true,
          "band": "HIGH",
          "model_id": "fraud_unsupervised_bifast_v3"
        }
      }
    }
  }]
}
```

---

## 7. End-to-End Request Lifecycle (Postman → Response)

1. **Postman** kirim POST ke `_ingest/pipeline/fraud_unsupervised_pipeline_v3_enriched/_simulate` dengan payload 23 field flat.
2. **Pre-request script** membungkus dokumen menjadi `{ "_source": { ... } }`.
3. **Elasticsearch** menerima request, menjalankan pipeline:
   - **Step 1 — enrich**: lookup `acct-velocity-snapshot` (yang di-maintain oleh continuous transform `acct_velocity_agg` setiap 1 menit dari `garda-context-bifast_trx`) untuk inject velocity fields per `sourceAccount`.
   - **Step 2 — painless preprocessing**: bangun engineered features (`log_amount`, `fee_ratio`, `cross_bank`, geo, hour-of-day, dst), Robust-scale numerik, one-hot encode kategori — pakai params dari `preprocess_params_v3.json`.
   - **Step 3 — inference**: panggil `_ml/trained_models/fraud_unsupervised_bifast_v3` (RF surrogate yang meniru IsolationForest), output → `ml.predicted_value`.
   - **Step 4 — band assignment**: set `ml.outlier_score`, `ml.is_anomaly`, dan `ml.band` berdasarkan threshold MEDIUM 0.6184 / HIGH 0.7309.
4. **Response** dikembalikan ke Postman dengan `ml.*` ter-attach ke `_source`.

---

## 8. Troubleshooting

| Masalah | Solusi |
|---|---|
| Latency call pertama 3000ms+ | Normal — Painless compile cost. Call ke-2 dst <300ms |
| `model_id not found` | `MODEL_VERSION=v3 python elastic/upload_surrogate.py` |
| `pipeline not found` | `python elastic/setup_v3_enriched.py` |
| Velocity fields semua 0 | Cek transform: `GET _transform/acct_velocity_agg/_stats` — pastikan state `started` dan `documents_processed > 0` |
| Score selalu LOW untuk akun baru | Akun belum ada di `acct-velocity-snapshot`; tunggu transform sync (maks ~1 menit) |
| Mau retrain | `python src/train_v3.py && MODEL_VERSION=v3 python elastic/upload_surrogate.py && python elastic/setup_v3_enriched.py` |

---

## 9. Versioning

| Version | Model | Pipeline | Status |
|---|---|---|---|
| v1 | `fraud_unsupervised_bifast` | `fraud_unsupervised_pipeline` | archived |
| v2 | `fraud_unsupervised_bifast_v2` | `fraud_unsupervised_pipeline_v2` | archived |
| **v3** | `fraud_unsupervised_bifast_v3` | `fraud_unsupervised_pipeline_v3_enriched` | **active** (with velocity enrichment + 3-band scoring) |
