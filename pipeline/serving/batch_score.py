from pathlib import Path
import joblib
import numpy as np
import pandas as pd
try:
    from pipeline.monitoring.metrics import get_batch_metrics
    MET = get_batch_metrics()
except Exception:
    MET = None

def batch_score(input_dir: str, output_dir: str, model_path: str, n_features: int) -> int:
    src = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_path)
    total = 0
    files = list(src.rglob('*.parquet'))
    for fp in files:
        df = pd.read_parquet(fp)
        cols = [f"f{i}" for i in range(n_features)]
        if not all(c in df.columns for c in cols):
            continue
        X = df[cols].to_numpy(dtype=float)
        preds = model.predict(X)
        res = pd.DataFrame({"prediction": preds})
        tgt = out / fp.relative_to(src)
        tgt.parent.mkdir(parents=True, exist_ok=True)
        res.to_parquet(tgt, index=False)
        total += len(res)
        if MET:
            MET['batch_files_processed'].inc()
            MET['batch_scored_rows'].inc(len(res))
    return total
