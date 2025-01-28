import numpy as np


class Histogram:
    def __init__(self, data):
        self.histogram = np.histogram(data)
