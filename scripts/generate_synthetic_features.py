from pathlib import Path
import numpy as np
import pandas as pd

def main():
    base = Path("ml-pipeline/runtime/features")
    base.mkdir(parents=True, exist_ok=True)
    n = 200
    n_features = 10
    data = {f"f{i}": np.random.rand(n) for i in range(n_features)}
    df = pd.DataFrame(data)
    df.to_parquet(base / "features.parquet", index=False)

if __name__ == "__main__":
    main()
