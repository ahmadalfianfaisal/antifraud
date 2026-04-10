# Setup ES Proxy

## Arsitektur

```
Java ──► Proxy (lab: 184.169.41.224:9200) ──► ES (184.169.41.143:9200)
  flat request      wrap _source                 _simulate OK
  clean response ◄── unwrap _source           ◄── ES response
```

Endpoint Java: `https://184.169.41.224:9200/_ingest/pipeline/.../_simulate`

## Step 1: SSH ke lab server

```bash
ssh 184.169.41.224
```

## Step 2: Install dependencies

```bash
pip3 install flask requests pyopenssl
```

## Step 3: Copy file proxy ke server

```bash
sudo mkdir -p /opt/es-proxy
sudo cp es_proxy.py /opt/es-proxy/
sudo cp es-proxy.service /etc/systemd/system/
```

## Step 4: Jalankan proxy

```bash
sudo systemctl daemon-reload
sudo systemctl enable es-proxy
sudo systemctl start es-proxy
```

Cek status:

```bash
sudo systemctl status es-proxy
```

## Step 5: Test

Request flat (tanpa `_source`):

```bash
curl -k -X POST \
  "https://184.169.41.224:9200/_ingest/pipeline/fraud_unsupervised_pipeline_v3_enriched/_simulate" \
  -H "Content-Type: application/json" \
  -d '{
  "docs": [
    {
      "bifastId": "TEST-001",
      "transactionDirection": "ORIGIN",
      "channel": "KOMIMOBILE",
      "transactionType": "CT",
      "sourceBic": "BMRIIDJA",
      "sourceAccount": "1599747454",
      "sourceAccountType": "SVGS",
      "sourceCountryCode": "",
      "destinationBic": "MEGAIDJA",
      "destinationAccount": "419109017",
      "destinationAccountType": "SVGS",
      "destinationCountryCode": "",
      "currency": "IDR",
      "amount": 15500,
      "fee": 0,
      "chargeType": "D",
      "chargeBearer": "DEBT",
      "countryCode": "ID",
      "deviceId": "SAMSUNG",
      "ipAddress": "10.10.20.15",
      "latitude": -6.171771400000,
      "longitude": 106.791342700000
    }
  ]
}'
```

Response akan flat (tanpa `_source`), langsung berisi data + `ml.band`, `ml.outlier_score`, `ml.is_anomaly`.

## Rollback

```bash
sudo systemctl stop es-proxy
sudo systemctl disable es-proxy
```
