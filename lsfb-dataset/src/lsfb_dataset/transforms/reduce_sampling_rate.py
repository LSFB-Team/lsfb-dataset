
class ReduceSamplingRate(object):
    def __init__(self, old_rate: int, new_rate: int):
        self.step = old_rate // new_rate

    def __call__(self, x):
        return x[::self.step]
