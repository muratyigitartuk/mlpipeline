from pathlib import Path
from datetime import datetime
from pipeline.ingestion.batch_ingest import generate_synthetic_orders, append_invalid_records
from pipeline.validation.validators import validate_parquet

def test_ingest_and_validate(tmp_path):
    raw_dir = tmp_path / "raw" / "orders"
    valid_dir = tmp_path / "curated" / "valid"
    invalid_dir = tmp_path / "curated" / "invalid"
    start = datetime.now().date().isoformat()
    total = generate_synthetic_orders(str(raw_dir), start, days=2, rows_per_day=50)
    append_invalid_records(str(raw_dir), start, 5)
    v, i = validate_parquet(str(raw_dir), str(valid_dir), str(invalid_dir))
    assert v + i == total + 5
    assert i > 0
