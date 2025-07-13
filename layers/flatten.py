from layers.base import Layer
import cupy as cp


class Flatten(Layer):
    """Flatten Layer Implementation
    This layer reshapes the input tensor to a 2D tensor by flattening all dimensions except the batch size.
    """

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
