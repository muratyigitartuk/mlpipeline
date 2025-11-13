from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.ingestion.batch_ingest import generate_synthetic_orders, append_invalid_records
from pipeline.validation.validators import validate_parquet

def main():
    base = Path("ml-pipeline/runtime")
    raw_dir = base / "raw" / "orders"
    valid_dir = base / "curated" / "valid"
    invalid_dir = base / "curated" / "invalid"
    start = datetime.now().date().isoformat()
    total = generate_synthetic_orders(str(raw_dir), start, days=2, rows_per_day=50)
    append_invalid_records(str(raw_dir), start, 5)
    v, i = validate_parquet(str(raw_dir), str(valid_dir), str(invalid_dir))
    print({"generated": total + 5, "valid": v, "invalid": i})

if __name__ == "__main__":
    main()
