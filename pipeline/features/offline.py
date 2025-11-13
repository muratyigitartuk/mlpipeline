from pathlib import Path
import pandas as pd

def compute_user_features(raw_orders_dir: str, output_parquet: str) -> str:
    parts = list(Path(raw_orders_dir).glob('date=*'))
    frames = []
    for p in parts:
        for fp in p.glob('*.parquet'):
            frames.append(pd.read_parquet(fp))
    if not frames:
        return output_parquet
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby('user_id').agg(order_count_30d=('user_id', 'count'), avg_order_amount_30d=('amount', 'mean')).reset_index()
    out = Path(output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(out, index=False)
    return str(out)
