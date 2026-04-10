"""Feature engineering + preprocessing untuk BI-FAST fraud detection (unsupervised)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

NUMERIC_RAW = ["amount", "fee", "latitude", "longitude"]
CATEGORICAL = [
    "channel", "transactionType", "deviceId",
    "sourceBic", "destinationBic",
    "sourceAccountType", "destinationAccountType",
    "chargeBearer", "chargeType",
]
ENGINEERED_NUM = [
    "log_amount", "log_fee", "fee_ratio",
    "has_geo", "has_device", "has_ip",
    "cross_bank", "cross_country",
]

# Aggregation features per sourceAccount over time windows.
# Aligned with rules: accumulative amount, velocity, device/ip/location switching, daily limit.
AGG_FEATURES = [
    "amt_sum_2min", "amt_sum_5min", "amt_sum_1h", "amt_sum_1d",
    "tx_count_5min", "tx_count_1h",
    "distinct_device_5min", "distinct_ip_5min", "distinct_location_5min",
]


def build_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung rolling aggregation per sourceAccount berdasarkan @timestamp.

    Cara pakai:
    - Training: panggil sekali di seluruh dataframe sebelum engineer().
    - Inference: caller menyediakan field AGG_FEATURES di payload (pre-computed
      dari state store / Elastic transform). Kalau tidak disediakan, default 0.
    """
    result = df.copy()
    for c in AGG_FEATURES:
        result[c] = 0.0

    if "@timestamp" not in df.columns or "sourceAccount" not in df.columns:
        return result

    tmp = pd.DataFrame({
        "orig_idx": df.index,
        "ts": pd.to_datetime(df["@timestamp"], errors="coerce", utc=True),
        "acc": df["sourceAccount"].fillna(-1).astype(str),
        "amt": pd.to_numeric(df.get("amount"), errors="coerce").fillna(0).values,
        "dev": df.get("deviceId", pd.Series("", index=df.index)).fillna("").astype(str).values,
        "ip": df.get("ipAddress", pd.Series("", index=df.index)).fillna("").astype(str).values,
    })
    lat = pd.to_numeric(df.get("latitude"), errors="coerce").round(2).astype(str)
    lon = pd.to_numeric(df.get("longitude"), errors="coerce").round(2).astype(str)
    tmp["loc"] = (lat + "_" + lon).values

    tmp = tmp.sort_values(["acc", "ts"]).reset_index(drop=True)

    agg = {c: np.zeros(len(tmp), dtype=float) for c in AGG_FEATURES}

    for _, grp in tmp.groupby("acc", sort=False):
        rows = grp.index.to_numpy()
        ts = grp["ts"].to_numpy()
        amt = grp["amt"].to_numpy()
        dev = grp["dev"].to_numpy()
        ip = grp["ip"].to_numpy()
        loc = grp["loc"].to_numpy()
        n = len(grp)
        j2 = j5 = jh = jd = 0
        for i in range(n):
            t = ts[i]
            if pd.isna(t):
                continue
            while j2 < i and (t - ts[j2]) > np.timedelta64(2, "m"):
                j2 += 1
            while j5 < i and (t - ts[j5]) > np.timedelta64(5, "m"):
                j5 += 1
            while jh < i and (t - ts[jh]) > np.timedelta64(1, "h"):
                jh += 1
            while jd < i and (t - ts[jd]) > np.timedelta64(1, "D"):
                jd += 1
            r = rows[i]
            agg["amt_sum_2min"][r] = amt[j2:i].sum()
            agg["amt_sum_5min"][r] = amt[j5:i].sum()
            agg["amt_sum_1h"][r] = amt[jh:i].sum()
            agg["amt_sum_1d"][r] = amt[jd:i].sum()
            agg["tx_count_5min"][r] = i - j5
            agg["tx_count_1h"][r] = i - jh
            agg["distinct_device_5min"][r] = len(set(dev[j5:i])) if i > j5 else 0
            agg["distinct_ip_5min"][r] = len(set(ip[j5:i])) if i > j5 else 0
            agg["distinct_location_5min"][r] = len(set(loc[j5:i])) if i > j5 else 0

    # map back to original order
    for c in AGG_FEATURES:
        tmp[c] = agg[c]
    tmp = tmp.sort_values("orig_idx")
    for c in AGG_FEATURES:
        result[c] = tmp[c].values
    return result


def engineer(df: pd.DataFrame, compute_agg: bool = False) -> pd.DataFrame:
    """Tambahkan derived features. Tidak mengubah df asli.

    compute_agg=True dipakai saat training untuk auto-compute rolling
    aggregation dari @timestamp. Saat inference, aggregation diharapkan
    sudah ada di payload (atau akan default 0).
    """
    if compute_agg:
        df = build_aggregations(df)
    out = df.copy()

    # ensure agg columns exist (default 0)
    for c in AGG_FEATURES:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    # numeric coercion
    for c in NUMERIC_RAW:
        out[c] = pd.to_numeric(out.get(c), errors="coerce")

    out["log_amount"] = np.log1p(out["amount"].fillna(0).clip(lower=0))
    out["log_fee"] = np.log1p(out["fee"].fillna(0).clip(lower=0))
    out["fee_ratio"] = (out["fee"].fillna(0) / out["amount"].replace(0, np.nan)).fillna(0)

    out["has_geo"] = out["latitude"].notna().astype(int)
    out["has_device"] = out.get("deviceId", pd.Series(index=out.index)).notna().astype(int)
    out["has_ip"] = out.get("ipAddress", pd.Series(index=out.index)).notna().astype(int)

    src_bic = out.get("sourceBic", pd.Series("", index=out.index)).fillna("")
    dst_bic = out.get("destinationBic", pd.Series("", index=out.index)).fillna("")
    out["cross_bank"] = (src_bic != dst_bic).astype(int)

    src_cc = out.get("sourceCountryCode", pd.Series("", index=out.index)).fillna("")
    dst_cc = out.get("destinationCountryCode", pd.Series("", index=out.index)).fillna("")
    out["cross_country"] = ((src_cc != dst_cc) & (src_cc != "") & (dst_cc != "")).astype(int)

    # ensure categorical columns exist as strings
    for c in CATEGORICAL:
        if c not in out.columns:
            out[c] = "UNKNOWN"
        out[c] = out[c].astype("object").fillna("UNKNOWN").replace("", "UNKNOWN")
    return out


def build_preprocessor() -> ColumnTransformer:
    num_cols = NUMERIC_RAW + ENGINEERED_NUM + AGG_FEATURES
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", cat_pipe, CATEGORICAL),
    ])
