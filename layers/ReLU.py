import cupy as cp
from layers.base import Layer


class ReLU(Layer):
    """
    ReLU Activation Layer
    Applies the rectified linear unit function element-wise: ReLU(x) = max(0, x)
    """

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask
