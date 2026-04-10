"""Orchestrator setup PoC v3 ENRICHED — fully ES-native fraud scoring.

Sequence:
  1. PUT enrich source index `acct-velocity-snapshot` (kalau belum ada — transform akan auto-create juga)
  2. PUT + START transform `acct_velocity_agg` (group garda-context-bifast_trx by sourceAccount)
  3. PUT + EXECUTE enrich policy `acct_velocity_policy`
  4. (Optional) Re-upload model `fraud_unsupervised_bifast_v3` via upload_surrogate.py
  5. PUT pipeline `fraud_unsupervised_pipeline_v3_enriched`
  6. Smoke test _simulate dengan satu payload sample

Run:
  python elastic/setup_v3_enriched.py            # full setup (skip model upload)
  python elastic/setup_v3_enriched.py --upload   # full setup + re-upload model
  python elastic/setup_v3_enriched.py --smoke    # smoke test only
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from base64 import b64encode

import requests

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELASTIC_DIR = os.path.join(ROOT, "elastic")

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

ES_HOST = os.environ["ES_HOST"]
ES_USER = os.environ["ES_USER"]
ES_PASS = os.environ["ES_PASS"]

TRANSFORM_ID = "acct_velocity_agg"
ENRICH_POLICY = "acct_velocity_policy"
PIPELINE_ID = "fraud_unsupervised_pipeline_v3_enriched"
MODEL_ID = "fraud_unsupervised_bifast_v3"

AUTH = "Basic " + b64encode(f"{ES_USER}:{ES_PASS}".encode()).decode()
HEADERS = {"Authorization": AUTH, "Content-Type": "application/json"}


def _req(method: str, path: str, body=None, params=None) -> requests.Response:
    url = f"{ES_HOST}{path}"
    r = requests.request(
        method, url, headers=HEADERS, json=body, params=params, verify=False, timeout=60
    )
    print(f"  [{method} {path}] {r.status_code}")
    if r.status_code >= 400:
        print(f"    {r.text[:500]}")
    return r


def load_json(filename: str) -> dict:
    with open(os.path.join(ELASTIC_DIR, filename)) as f:
        return json.load(f)


def step_transform():
    print("\n[1/5] Transform acct_velocity_agg")
    body = load_json("transform_acct_velocity.json")
    # delete first jika exist (idempotent)
    _req("POST", f"/_transform/{TRANSFORM_ID}/_stop", params={"force": "true"})
    _req("DELETE", f"/_transform/{TRANSFORM_ID}", params={"force": "true"})
    r = _req("PUT", f"/_transform/{TRANSFORM_ID}", body=body)
    if r.status_code >= 400:
        return False
    _req("POST", f"/_transform/{TRANSFORM_ID}/_start")
    print("  waiting 5s for transform first sync...")
    time.sleep(5)
    _req("GET", f"/_transform/{TRANSFORM_ID}/_stats")
    return True


def step_enrich_policy():
    print("\n[2/5] Enrich policy acct_velocity_policy")
    body = load_json("enrich_policy_acct_velocity.json")
    _req("DELETE", f"/_enrich/policy/{ENRICH_POLICY}")
    r = _req("PUT", f"/_enrich/policy/{ENRICH_POLICY}", body=body)
    if r.status_code >= 400:
        return False
    _req("POST", f"/_enrich/policy/{ENRICH_POLICY}/_execute")
    return True


def step_model_upload():
    print("\n[3/5] Re-upload model fraud_unsupervised_bifast_v3 (via upload_surrogate.py)")
    os.environ["MODEL_VERSION"] = "v3"
    upload_path = os.path.join(ELASTIC_DIR, "upload_surrogate.py")
    code = os.system(f'"{sys.executable}" "{upload_path}"')
    return code == 0


def step_pipeline():
    print("\n[4/5] Pipeline fraud_unsupervised_pipeline_v3_enriched")
    body = load_json("fraud_unsupervised_pipeline_v3_enriched.json")
    r = _req("PUT", f"/_ingest/pipeline/{PIPELINE_ID}", body=body)
    return r.status_code < 400


def step_smoke_test():
    print("\n[5/5] Smoke test _simulate")
    payload = {
        "docs": [
            {
                "_source": {
                    "bifastId": "SMOKE-HIGH-001",
                    "transactionId": "smoke-001",
                    "transactionDirection": "ORIGIN",
                    "channel": "KOMIMOBILE",
                    "transactionType": "MBANKING",
                    "sourceBic": "BMRIIDJA",
                    "sourceAccount": "8831239430",
                    "sourceAccountType": "SVGS",
                    "sourceCountryCode": "",
                    "destinationBic": "MEGAIDJA",
                    "destinationAccount": "7890254163",
                    "destinationAccountType": "SVGS",
                    "destinationCountryCode": "",
                    "currency": "IDR",
                    "amount": 500000000000.0,
                    "fee": 0.0,
                    "chargeType": "D",
                    "chargeBearer": "DEBT",
                    "countryCode": "ID",
                    "deviceId": "SAMSUNG",
                    "ipAddress": "10.10.20.15",
                    "latitude": -6.1717714,
                    "longitude": 106.7913427,
                }
            }
        ]
    }
    t0 = time.time()
    r = _req("POST", f"/_ingest/pipeline/{PIPELINE_ID}/_simulate", body=payload)
    elapsed = (time.time() - t0) * 1000
    print(f"  latency: {elapsed:.1f} ms")
    if r.status_code < 400:
        out = r.json()
        ml = out.get("docs", [{}])[0].get("doc", {}).get("_source", {}).get("ml", {})
        print(f"  ml: {json.dumps(ml, indent=2)}")
    return r.status_code < 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="re-upload model juga")
    ap.add_argument("--smoke", action="store_true", help="smoke test only")
    args = ap.parse_args()

    print(f"Target ES: {ES_HOST}")

    if args.smoke:
        step_smoke_test()
        return

    if not step_transform():
        print("ABORT: transform failed")
        return
    if not step_enrich_policy():
        print("ABORT: enrich policy failed")
        return
    if args.upload:
        if not step_model_upload():
            print("WARN: model upload failed, lanjut ke pipeline (asumsi model sudah ada)")
    if not step_pipeline():
        print("ABORT: pipeline failed")
        return
    step_smoke_test()
    print("\nDONE.")


if __name__ == "__main__":
    main()
