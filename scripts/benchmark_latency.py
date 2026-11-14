from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import time
import numpy as np
from fastapi.testclient import TestClient

from needle.api import create_app


def main() -> None:
    app = create_app(seed_history=False)
    t0 = datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc)
    latencies_ms: list[float] = []

    with TestClient(app) as client:
        for i in range(20):
            client.post(
                "/score",
                json={
                    "txn_id": f"WARM_{i}",
                    "card_id": f"CARD_W_{i % 10}",
                    "ts": (t0 + timedelta(seconds=i)).isoformat(),
                    "amount": 1200.0,
                    "merchant_id": "M_5411_100",
                    "mcc": 5411,
                    "channel": "POS",
                    "country": "IN",
                    "device_id": None,
                },
            )

        for i in range(1000):
            payload = {
                "txn_id": f"BENCH_{i:04d}",
                "card_id": f"CARD_B_{i % 50}",
                "ts": (t0 + timedelta(minutes=5, seconds=i)).isoformat(),
                "amount": 1500.0 + (i % 10) * 250.0,
                "merchant_id": "M_5411_101",
                "mcc": 5411,
                "channel": "POS",
                "country": "IN",
                "device_id": None,
            }
            start = time.perf_counter()
            resp = client.post("/score", json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            assert resp.status_code == 200
            latencies_ms.append(elapsed_ms)

    arr = np.array(latencies_ms)
    summary = {
        "requests": 1000,
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
