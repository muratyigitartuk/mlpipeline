from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def run_drift(reference_path: str, current_path: str, output_html: str) -> str:
    ref = pd.read_parquet(reference_path)
    cur = pd.read_parquet(current_path)
    r = Report(metrics=[DataDriftPreset()])
    r.run(reference_data=ref, current_data=cur)
    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    r.save(str(out))
    return str(out)
