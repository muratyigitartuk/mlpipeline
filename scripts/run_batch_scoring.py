from pathlib import Path
import yaml
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.serving.batch_score import batch_score

def main():
    cfg_path = Path("config.yaml")
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        n_features = int(cfg.get("model", {}).get("n_features", 10))
        model_path = cfg.get("registry", {}).get("local_production_path") or cfg.get("model", {}).get("model_path")
        mp = Path(model_path)
        if not mp.is_absolute():
            if not mp.exists():
                mp = Path("ml-pipeline") / mp
        model_path = str(mp)
    else:
        n_features = 10
        model_path = str(Path("ml-pipeline") / "model" / "model.joblib")
    inp = "ml-pipeline/runtime/features"
    out = "ml-pipeline/runtime/scored"
    total = batch_score(inp, out, model_path, n_features)
    print({"scored": total, "output_dir": out})

if __name__ == "__main__":
    main()
