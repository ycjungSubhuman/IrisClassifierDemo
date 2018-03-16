import numpy as np

def normalize(d, minval, maxval):
    m = (minval + maxval) / 2
    result = (d - m) / ((maxval - minval) * 2)
    return result

def dist(a, b):
    delta = a-b
    return np.sqrt(delta.dot(delta))

