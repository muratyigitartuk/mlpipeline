from pathlib import Path
import yaml
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.training.train import train_logistic, train_with_mlflow

def main():
    feats = Path("ml-pipeline/runtime/features/features.parquet")
    out = Path("ml-pipeline/model/model.joblib")
    cfg_path = Path("config.yaml")
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        mlf = cfg.get("mlflow", {})
        uri = mlf.get("tracking_uri")
        exp = mlf.get("experiment")
    else:
        uri = None
        exp = None
    p = out
    if uri:
        train_with_mlflow(str(feats), str(p), uri, exp)
    else:
        train_logistic(str(feats), str(p))
    print(str(p))

if __name__ == "__main__":
    main()
