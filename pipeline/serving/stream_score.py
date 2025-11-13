import numpy as np

def stream_score(model, messages):
    outputs = []
    for msg in messages:
        X = np.array([msg])
        y = model.predict(X)
        outputs.append(float(y[0]) if hasattr(y, '__iter__') else float(y))
    return outputs
