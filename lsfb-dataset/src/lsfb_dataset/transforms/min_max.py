import numpy as np
from sklearn.preprocessing import MinMaxScaler


class MinMaxRescale(object):

    def __call__(self, features: np.ndarray):
        scaler = MinMaxScaler(copy=False)
        return scaler.fit_transform(features)
