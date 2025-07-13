import cupy as cp
from layers.base import Layer


class Dropout(Layer):
    def __init__(self, drop_prob=0.5):
        self.drop_prob = drop_prob
        self.mask = None
        self.training = True

    def forward(self, x):
        if self.training:

            keep_prob = 1 - self.drop_prob
            self.mask = (cp.random.rand(*x.shape) < keep_prob) / keep_prob
            return x * self.mask
        else:

            return x

    def backward(self, grad_output):

        if self.training and self.mask is not None:
            return grad_output * self.mask
        else:
            return grad_output
