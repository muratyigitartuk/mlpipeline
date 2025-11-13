from prometheus_client import Counter

def get_batch_metrics():
    return {
        'batch_scored_rows': Counter('batch_scored_rows', 'Total rows scored in batch'),
        'batch_files_processed': Counter('batch_files_processed', 'Total files processed in batch'),
    }
