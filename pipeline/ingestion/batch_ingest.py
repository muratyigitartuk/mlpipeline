from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def generate_synthetic_orders(output_dir: str, start_date: str, days: int, rows_per_day: int) -> int:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = datetime.fromisoformat(start_date)
    total = 0
    for i in range(days):
        day = start + timedelta(days=i)
        ts = pd.to_datetime(day.date())
        user_id = np.random.randint(1, 1000, size=rows_per_day)
        amount = np.round(np.random.rand(rows_per_day) * 1000, 2)
        product_id = np.random.randint(1, 500, size=rows_per_day)
        event_time = pd.to_datetime(ts) + pd.to_timedelta(np.random.randint(0, 86400, rows_per_day), unit="s")
        df = pd.DataFrame({
            "user_id": user_id,
            "amount": amount,
            "product_id": product_id,
            "event_time": event_time,
        })
        part_dir = out / f"date={ts.date()}"
        part_dir.mkdir(parents=True, exist_ok=True)
        file_path = part_dir / "orders.parquet"
        df.to_parquet(file_path, index=False)
        total += len(df)
    return total

def append_invalid_records(output_dir: str, date: str, count: int) -> int:
    out = Path(output_dir) / f"date={date}"
    out.mkdir(parents=True, exist_ok=True)
    user_id = np.random.randint(1, 1000, size=count)
    amount = -np.abs(np.round(np.random.rand(count) * 1000, 2))
    product_id = np.random.randint(1, 500, size=count)
    event_time = pd.to_datetime(date) + pd.to_timedelta(np.random.randint(0, 86400, count), unit="s")
    df = pd.DataFrame({
        "user_id": user_id,
        "amount": amount,
        "product_id": product_id,
        "event_time": event_time,
    })
    file_path = out / "orders_extra.parquet"
    df.to_parquet(file_path, index=False)
    return len(df)
