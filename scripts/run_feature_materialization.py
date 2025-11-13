from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.features.offline import compute_user_features
from pipeline.features.online import materialize_to_sqlite, get_features_sqlite

def main():
    raw_dir = Path('ml-pipeline/runtime/raw/orders')
    offline = Path('ml-pipeline/runtime/features_agg/user_features.parquet')
    online = Path('ml-pipeline/runtime/online/features.db')
    compute_user_features(str(raw_dir), str(offline))
    materialize_to_sqlite(str(offline), str(online))
    sample = get_features_sqlite(str(online), 1)
    print(sample)

if __name__ == '__main__':
    main()
