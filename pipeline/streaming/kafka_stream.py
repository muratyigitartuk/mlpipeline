import json
from pathlib import Path
import joblib
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

def consume_and_score(bootstrap_servers: str, input_topic: str, output_topic: str, group_id: str, model_path: str):
    model = joblib.load(model_path)
    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=True,
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    )
    for msg in consumer:
        payload = msg.value
        feats = np.array([payload.get('features')], dtype=float)
        pred = model.predict(feats)
        out = {'prediction': float(pred[0])}
        producer.send(output_topic, out)
        producer.flush()

