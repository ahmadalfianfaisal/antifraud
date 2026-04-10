"""Train IsolationForest v3 — fitur velocity + threshold dikalibrasi pakai Rules GARDA.

Beda dari train.py:
- Fit ulang iforest pakai feature set v3 (12 num lama + 9 agg velocity).
- Threshold tidak pakai P95 buta — pakai pseudo-label dari Rules GARDA lalu
  pilih titik dengan F1 maksimum di precision-recall curve.
- Tambah band HIGH / MEDIUM supaya score borderline (mis. 0.63) tidak hilang.
- Simpan artefak terpisah: fraud_iforest_v3.joblib + model_meta_v3.json.
"""
from __future__ import annotations
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve

from preprocessing import engineer, build_preprocessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(ROOT, "data", "bifast_trx.csv")
MODELS_DIR = os.path.join(ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "fraud_iforest_v3.joblib")
META_PATH = os.path.join(MODELS_DIR, "model_meta_v3.json")
SCORED_CSV = os.path.join(ROOT, "data", "bifast_trx_scored_v3.csv")

CONTAMINATION = 0.05
FALLBACK_PERCENTILE = 95   # dipakai kalau rule_hit kosong

# Threshold rules GARDA (lihat Rules Garda - Komi.xlsx).
# Angka di-set konservatif — tweak sesuai kebijakan bisnis.
RULE_AMT_2MIN = 350_000_000
RULE_AMT_5MIN = 100_000_000      # batas terkecil (Blue=25jt s/d Gold=100jt)
RULE_AMT_1D = 150_000_000        # daily limit terkecil (Blue)
RULE_TX_5MIN = 3                 # minimum interval (Blue=3x)
RULE_TX_1H = 51
RULE_DISTINCT_5MIN = 2           # device/ip/location switching


def normalize_score(raw: np.ndarray) -> np.ndarray:
    s = -raw
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def apply_garda_rules(feats: pd.DataFrame) -> np.ndarray:
    """Pseudo-label: 1 kalau transaksi melanggar minimal satu rule GARDA."""
    g = (
        (feats["amt_sum_2min"] >= RULE_AMT_2MIN)
        | (feats["amt_sum_5min"] >= RULE_AMT_5MIN)
        | (feats["amt_sum_1d"] >= RULE_AMT_1D)
        | (feats["tx_count_5min"] >= RULE_TX_5MIN)
        | (feats["tx_count_1h"] >= RULE_TX_1H)
        | (feats["distinct_device_5min"] >= RULE_DISTINCT_5MIN)
        | (feats["distinct_ip_5min"] >= RULE_DISTINCT_5MIN)
        | (feats["distinct_location_5min"] >= RULE_DISTINCT_5MIN)
    )
    return g.astype(int).to_numpy()


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Pilih threshold yang maksimalkan F1 vs pseudo-label rules GARDA.

    Fallback ke persentil 95 kalau tidak ada hit (rule belum matching apapun).
    """
    n_pos = int(labels.sum())
    if n_pos == 0:
        thr = float(np.percentile(scores, FALLBACK_PERCENTILE))
        return {"threshold": thr, "method": "fallback_p95", "f1": None,
                "precision": None, "recall": None, "n_pos_label": 0}

    prec, rec, thr_arr = precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    # precision_recall_curve: prec/rec length = len(thr)+1
    best = int(np.argmax(f1[:-1])) if len(thr_arr) else 0
    return {
        "threshold": float(thr_arr[best]) if len(thr_arr) else 0.5,
        "method": "f1_max_vs_garda_rules",
        "f1": float(f1[best]),
        "precision": float(prec[best]),
        "recall": float(rec[best]),
        "n_pos_label": n_pos,
    }


def main():
    print(f"Loading {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    print(f"  rows={len(df)}  cols={df.shape[1]}")

    feats = engineer(df, compute_agg=True)
    pre = build_preprocessor()

    pipe = Pipeline([
        ("pre", pre),
        ("iforest", IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination=CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Fitting pipeline (v3)...")
    pipe.fit(feats)

    raw = pipe.named_steps["iforest"].score_samples(
        pipe.named_steps["pre"].transform(feats)
    )
    raw_min, raw_max = float(raw.min()), float(raw.max())
    scores = normalize_score(raw)

    # ----- threshold: percentile-based, dengan info kalibrasi GARDA -----
    rule_hit = apply_garda_rules(feats)
    cal = calibrate_threshold(scores, rule_hit)
    # MEDIUM = P90 (~10% review), HIGH = P98 (~2% auto-block)
    threshold = float(np.percentile(scores, 90))
    threshold_high = float(np.percentile(scores, 98))

    is_outlier = (scores >= threshold).astype(int)

    print(f"score range: {scores.min():.3f} - {scores.max():.3f}")
    print(f"rule_hit count : {int(rule_hit.sum())} / {len(rule_hit)}")
    print(f"calibration    : {cal}")
    print(f"threshold (MED): {threshold:.4f}")
    print(f"threshold (HIGH): {threshold_high:.4f}")
    print(f"flagged outliers (>=MED): {is_outlier.sum()} / {len(is_outlier)}")

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

    meta = {
        "model_id": "fraud_unsupervised_bifast_v3",
        "algorithm": "IsolationForest",
        "feature_set": "v3 (base + velocity aggregations)",
        "contamination": CONTAMINATION,
        "score_normalization": "neg_score min-max -> [0,1]",
        "threshold": threshold,
        "threshold_high": threshold_high,
        "threshold_calibration": cal,
        "bands": {
            "HIGH": f">= {threshold_high:.4f}  (auto-block)",
            "MEDIUM": f">= {threshold:.4f} and < {threshold_high:.4f}  (review / step-up)",
            "LOW": f"< {threshold:.4f}  (pass)",
        },
        "n_train": int(len(feats)),
        "n_outliers_med": int(is_outlier.sum()),
        "n_outliers_high": int((scores >= threshold_high).sum()),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "rules_used": {
            "amt_sum_2min>=": RULE_AMT_2MIN,
            "amt_sum_5min>=": RULE_AMT_5MIN,
            "amt_sum_1d>=": RULE_AMT_1D,
            "tx_count_5min>=": RULE_TX_5MIN,
            "tx_count_1h>=": RULE_TX_1H,
            "distinct_*_5min>=": RULE_DISTINCT_5MIN,
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta  -> {META_PATH}")

    scored = df.copy()
    scored["outlier_score"] = scores
    scored["rule_hit"] = rule_hit
    scored["band"] = np.where(
        scores >= threshold_high, "HIGH",
        np.where(scores >= threshold, "MEDIUM", "LOW"),
    )
    scored.to_csv(SCORED_CSV, index=False)
    print(f"Saved scored CSV -> {SCORED_CSV}")

    print("\nTop-10 outliers (v3):")
    cols = ["bifastId", "amount", "channel", "sourceBic", "destinationBic",
            "deviceId", "outlier_score", "band", "rule_hit"]
    cols = [c for c in cols if c in scored.columns]
    print(scored.sort_values("outlier_score", ascending=False)[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
