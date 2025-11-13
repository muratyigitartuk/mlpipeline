from pathlib import Path
from typing import Tuple
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from datetime import datetime

class OrderRecord(BaseModel):
    model_config = ConfigDict(extra='ignore')
    user_id: int = Field(gt=0)
    amount: float = Field(ge=0, le=100000)
    product_id: int = Field(gt=0)
    event_time: datetime

def validate_parquet(input_dir: str, output_valid_dir: str, output_invalid_dir: str) -> Tuple[int, int]:
    src = Path(input_dir)
    ok = Path(output_valid_dir)
    bad = Path(output_invalid_dir)
    ok.mkdir(parents=True, exist_ok=True)
    bad.mkdir(parents=True, exist_ok=True)
    total_valid = 0
    total_invalid = 0
    for part in sorted(p for p in src.glob('date=*') if p.is_dir()):
        frames = []
        for fp in part.glob('*.parquet'):
            frames.append(pd.read_parquet(fp))
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        valid_rows = []
        invalid_rows = []
        for _, row in df.iterrows():
            try:
                OrderRecord(**row.to_dict())
                valid_rows.append(row)
            except ValidationError:
                invalid_rows.append(row)
        if valid_rows:
            vdf = pd.DataFrame(valid_rows)
            tgt = ok / part.name
            tgt.mkdir(parents=True, exist_ok=True)
            vdf.to_parquet(tgt / 'valid.parquet', index=False)
            total_valid += len(vdf)
        if invalid_rows:
            idf = pd.DataFrame(invalid_rows)
            tgt = bad / part.name
            tgt.mkdir(parents=True, exist_ok=True)
            idf.to_parquet(tgt / 'invalid.parquet', index=False)
            total_invalid += len(idf)
    return total_valid, total_invalid
