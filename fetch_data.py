import os
import json
import warnings
from base64 import b64encode

import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

warnings.filterwarnings('ignore', category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

# === Config ===
ES_HOST = os.environ['ES_HOST']
INDEX = os.environ.get('ES_INDEX', 'garda-context-bifast_trx')
USERNAME = os.environ['ES_USER']
PASSWORD = os.environ['ES_PASS']

OUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(OUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUT_DIR, 'bifast_trx.csv')
OUTPUT_JSON = os.path.join(OUT_DIR, 'bifast_trx_raw.json')

SEARCH_URL = f'{ES_HOST}/{INDEX}/_search?scroll=2m'
SCROLL_URL = f'{ES_HOST}/_search/scroll'

auth_b64 = b64encode(f'{USERNAME}:{PASSWORD}'.encode()).decode()
HEADERS = {
    'Content-Type': 'application/json',
    'kbn-xsrf': 'true',
    'Authorization': f'Basic {auth_b64}',
}

payload = {"size": 1000, "query": {"match_all": {}}}


def clear_scroll(scroll_id, session):
    try:
        r = session.delete(SCROLL_URL, headers=HEADERS, json={"scroll_id": scroll_id}, verify=False)
        r.raise_for_status()
        print("Scroll context cleared")
    except requests.exceptions.RequestException as e:
        print(f"Error clearing scroll: {e}")


def get_elasticsearch_data():
    session = requests.Session()
    scroll_id = None
    try:
        r = session.get(SEARCH_URL, headers=HEADERS, json=payload, verify=False)
        r.raise_for_status()
        data = r.json()

        hits = data.get('hits', {}).get('hits', [])
        scroll_id = data.get('_scroll_id')
        total = data.get('hits', {}).get('total', {}).get('value', 0)

        if not scroll_id or not hits:
            print("No scroll_id / no hits")
            return pd.DataFrame()

        records = [h['_source'] for h in hits]

        with tqdm(total=total, desc="Fetching", unit="rec") as pbar:
            pbar.update(len(hits))
            while hits:
                r = session.post(
                    SCROLL_URL,
                    headers=HEADERS,
                    json={"scroll": "2m", "scroll_id": scroll_id},
                    verify=False,
                )
                r.raise_for_status()
                data = r.json()
                hits = data.get('hits', {}).get('hits', [])
                scroll_id = data.get('_scroll_id', scroll_id)
                records.extend(h['_source'] for h in hits)
                pbar.update(len(hits))

        return pd.DataFrame(records)

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return pd.DataFrame()
    finally:
        if scroll_id:
            clear_scroll(scroll_id, session)
        session.close()


def save_outputs(df: pd.DataFrame):
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Saved CSV  -> {OUTPUT_CSV}  ({len(df)} rows)")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved JSON -> {OUTPUT_JSON}")


if __name__ == "__main__":
    df = get_elasticsearch_data()
    if not df.empty:
        print("\nSample:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())
        save_outputs(df)
    else:
        print("No data retrieved")
