from functools import partial
import pandas as pd

lr = 1e-6
conv_condition = 1e-8


def predicted_value(w, b, x):
    return w * x + b


def extract_params(data):
    w, b = 0, 0
    nw, nb = calculate_params(w, b, data)
    while abs(nw - w) > conv_condition and abs(nb - b) > conv_condition:
        w, b = nw, nb
        nw, nb = calculate_params(w, b, data)
    return w, b


def calculate_params(w, b, data):
    pvf = partial(predicted_value, w, b)
    pv = pvf(data[:, 0])
    av = data[:, 1]
    diff = pv - av
    nw = w - lr * (1 / len(data)) * sum(diff * data[:, 0])
    nb = b - lr * (1 / len(data)) * sum(diff)
    return nw, nb


data = pd.read_csv("realest.csv", header=0)
space_price = data[["Space", "Price"]].dropna().to_numpy()

print(extract_params(space_price))
