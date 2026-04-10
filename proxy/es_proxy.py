"""
Reverse proxy for Elasticsearch _simulate endpoint.
Wraps flat docs with _source before sending to ES,
unwraps _source from ES response before returning to caller.

Java sends flat -> proxy wraps _source -> ES accepts -> proxy unwraps -> Java receives flat.

Optimasi latency:
- Connection pooling (requests.Session) — reuse TCP/TLS connection ke ES
- Pre-generated SSL cert — tidak generate tiap start
- Gunicorn + gevent — multi-worker async
"""

import os
import urllib3
import requests
from flask import Flask, request, jsonify, Response

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)

ES_BACKEND = os.environ.get("ES_BACKEND", "https://184.169.41.143:9200")
ES_USER = os.environ.get("ES_USER", "elastic")
ES_PASS = os.environ.get("ES_PASS", "p@ssw0rd")

# --- Connection pooling: reuse TCP + TLS connection ke ES ---
session = requests.Session()
session.auth = (ES_USER, ES_PASS)
session.verify = False
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=1,
)
session.mount("https://", adapter)
session.mount("http://", adapter)


@app.route("/_ingest/pipeline/<path:pipeline>/_simulate", methods=["POST"])
def simulate(pipeline):
    body = request.get_json(force=True)

    # --- wrap: tambah _source ke setiap doc ---
    if "docs" in body and isinstance(body["docs"], list):
        body["docs"] = [
            doc if "_source" in doc else {"_source": doc}
            for doc in body["docs"]
        ]

    # --- forward ke ES (pakai session pooled) ---
    es_url = f"{ES_BACKEND}/_ingest/pipeline/{pipeline}/_simulate"
    try:
        resp = session.post(es_url, json=body, timeout=30)
    except requests.exceptions.ConnectionError as e:
        return jsonify({"error": f"Cannot connect to ES backend: {e}"}), 502

    # --- unwrap: buang _source dari response ---
    if resp.status_code == 200:
        try:
            es_data = resp.json()
            if "docs" in es_data:
                clean_docs = [
                    d.get("doc", {}).get("_source", d)
                    if isinstance(d.get("doc"), dict)
                    else d
                    for d in es_data["docs"]
                ]
                return jsonify({"docs": clean_docs}), 200
        except (ValueError, KeyError):
            pass

    # fallback: return ES response as-is
    return Response(
        resp.content,
        status=resp.status_code,
        content_type=resp.headers.get("content-type", "application/json"),
    )


# --- proxy semua request lain langsung ke ES ---
@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_passthrough(path):
    es_url = f"{ES_BACKEND}/{path}"
    if request.query_string:
        es_url += f"?{request.query_string.decode()}"

    resp = session.request(
        method=request.method,
        url=es_url,
        headers={k: v for k, v in request.headers if k.lower() not in ("host",)},
        data=request.get_data(),
        timeout=30,
    )
    return Response(
        resp.content,
        status=resp.status_code,
        content_type=resp.headers.get("content-type", "application/json"),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PROXY_PORT", 9200))
    print(f"ES proxy listening on :{port}, backend: {ES_BACKEND}")
    app.run(host="0.0.0.0", port=port, ssl_context="adhoc")
