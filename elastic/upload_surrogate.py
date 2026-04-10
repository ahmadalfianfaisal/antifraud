"""Train surrogate RandomForestRegressor yang meniru IsolationForest score,
lalu upload ke Elasticsearch via eland.

Surrogate dipakai karena Elastic _ml/trained_models tidak natively support
IsolationForest, tapi mendukung tree ensemble regression.
"""
from __future__ import annotations
import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from preprocessing import engineer  # noqa: E402

DATA_CSV = os.path.join(ROOT, "data", "bifast_trx.csv")
VERSION = os.environ.get("MODEL_VERSION", "v2")
_IFOREST_FILE = "fraud_iforest_v3.joblib" if VERSION == "v3" else "fraud_iforest.joblib"
_META_FILE = "model_meta_v3.json" if VERSION == "v3" else "model_meta.json"
MODEL_PATH = os.path.join(ROOT, "models", _IFOREST_FILE)
META_PATH = os.path.join(ROOT, "models", _META_FILE)
SURROGATE_PATH = os.path.join(ROOT, "models", f"surrogate_rf_{VERSION}.joblib")

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

ES_HOST = os.environ["ES_HOST"]
ES_USER = os.environ["ES_USER"]
ES_PASS = os.environ["ES_PASS"]
MODEL_ID = f"fraud_unsupervised_bifast_{VERSION}"

# slim params for v2: smaller forest, faster inference
RF_PARAMS = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 4}


def sanitize(name: str) -> str:
    return name.replace("[", "_").replace("]", "_").replace(" ", "_").replace("'", "")


def main():
    print("Loading IForest pipeline + data...")
    pipe = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    df = pd.read_csv(DATA_CSV)
    feats = engineer(df, compute_agg=True)

    pre = pipe.named_steps["pre"]
    iforest = pipe.named_steps["iforest"]

    X = pre.transform(feats)
    raw = iforest.score_samples(X)
    s = -raw
    lo, hi = float(s.min()), float(s.max())
    y = np.clip((s - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(s)

    feature_names = [sanitize(n) for n in pre.get_feature_names_out().tolist()]
    print(f"X shape={X.shape}  features={len(feature_names)}")

    print("Training surrogate RandomForestRegressor...")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, **RF_PARAMS)
    rf.fit(X, y)
    pred = rf.predict(X)
    mae = float(np.mean(np.abs(pred - y)))
    corr = float(np.corrcoef(pred, y)[0, 1])
    print(f"Surrogate fit: MAE={mae:.4f}  corr={corr:.4f}")

    joblib.dump({"rf": rf, "feature_names": feature_names}, SURROGATE_PATH)
    print(f"Saved surrogate -> {SURROGATE_PATH}")

    # Upload via eland
    try:
        from elasticsearch import Elasticsearch
        from eland.ml import MLModel
        es = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASS), verify_certs=False)
        print(f"ES: {es.info()['version']['number']}")
        MLModel.import_model(
            es_client=es,
            model_id=MODEL_ID,
            model=rf,
            feature_names=feature_names,
            es_if_exists="replace",
        )
        print(f"Uploaded model_id={MODEL_ID}")
    except Exception as e:
        print(f"Upload skipped: {e}")
        print("Run later: python elastic/upload_surrogate.py")


if __name__ == "__main__":
    main()
