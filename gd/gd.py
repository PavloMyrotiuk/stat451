from functools import partial
import numpy as np
from functools import reduce

lr = 1e-3
conv_condition = 0.01


def predicted_value(w, b, x):
    return np.sum(x * w + b, axis=1).reshape(-1, 1)


def extract_params(data):
    w, b = np.zeros(data.shape[1] - 1), 0
    nw, nb = calculate_params(w, b, data)
    while reduce(lambda k, l: k or l, nw - w > conv_condition) or abs(nb - b) > conv_condition:
        w, b = nw, nb
        nw, nb = calculate_params(w, b, data)
    return w, b


def calculate_params(w, b, data):
    pvf = partial(predicted_value, w, b)
    x = data[:, :-1]
    pv = pvf(x)
    av = data[:, -1:]
    diff = pv - av
    nw = w - lr * (1 / len(data)) * np.sum(diff * x, axis=0)
    nb = b - lr * (1 / len(data)) * np.sum(diff)
    return nw, nb
