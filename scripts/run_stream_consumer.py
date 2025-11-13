import json
from pathlib import Path
import joblib
import numpy as np

def main():
    model = joblib.load(str(Path("ml-pipeline/model/model.joblib")))
    inp = Path("ml-pipeline/runtime/stream/input.jsonl")
    out = Path("ml-pipeline/runtime/stream/output.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with inp.open() as fi, out.open("w") as fo:
        for line in fi:
            msg = json.loads(line)
            X = np.array([msg["features"]], dtype=float)
            y = model.predict(X)
            fo.write(json.dumps({"prediction": float(y[0])}) + "\n")

if __name__ == "__main__":
    main()
