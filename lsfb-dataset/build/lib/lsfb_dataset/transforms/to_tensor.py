import torch


class ToTensor(object):

    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, array):
        array = torch.from_numpy(array)

        if self.dtype is not None:
            array = array.type(self.dtype)

        return array
