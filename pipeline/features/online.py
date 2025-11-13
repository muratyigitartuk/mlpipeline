from pathlib import Path
import sqlite3
import pandas as pd

def materialize_to_sqlite(features_parquet: str, sqlite_path: str) -> str:
    df = pd.read_parquet(features_parquet)
    db = Path(sqlite_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    df.to_sql('features', conn, if_exists='replace', index=False)
    conn.close()
    return str(db)

def get_features_sqlite(sqlite_path: str, user_id: int) -> dict:
    conn = sqlite3.connect(sqlite_path)
    q = pd.read_sql_query('select * from features where user_id = ?', conn, params=[user_id])
    conn.close()
    return q.iloc[0].to_dict() if not q.empty else {}
