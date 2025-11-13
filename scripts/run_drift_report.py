from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.monitoring.drift_job import run_drift

def main():
    base = Path("ml-pipeline/runtime/features")
    ref = base / "features.parquet"
    cur = base / "features.parquet"
    out = Path("ml-pipeline/runtime/reports/drift.html")
    p = run_drift(str(ref), str(cur), str(out))
    print(p)

if __name__ == "__main__":
    main()
