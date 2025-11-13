from pathlib import Path
import yaml
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.streaming.kafka_stream import consume_and_score

def main():
    cfg_path = Path('config.yaml')
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    k = cfg.get('streaming', {}).get('kafka', {})
    bs = k.get('bootstrap_servers', 'localhost:9092')
    it = k.get('input_topic', 'inference_requests')
    ot = k.get('output_topic', 'inference_responses')
    gid = k.get('group_id', 'mlpipeline-consumer')
    mp = cfg.get('registry', {}).get('local_production_path') or cfg.get('model', {}).get('model_path')
    mp = str((Path('ml-pipeline') / Path(mp)).resolve()) if mp and not Path(mp).is_absolute() else mp
    consume_and_score(bs, it, ot, gid, mp)

if __name__ == '__main__':
    main()
