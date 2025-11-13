from pathlib import Path
import shutil

def main():
    src = Path("ml-pipeline/model/model.joblib")
    dst = Path("ml-pipeline/model/production/model.joblib")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(str(dst))

if __name__ == "__main__":
    main()
