"""
Dense Layer Implementation
"""

from layers.base import Layer
from layers.backend import xp


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.w = xp.random.randn(input_size, output_size) * 0.01
        self.b = xp.zeros(output_size)

    def forward(self, x):
        """
        Forward pass of the dense layer.

        Args:
            x (cupy.ndarray): Input tensor of shape (N, input_size).

        Returns:
            cupy.ndarray: Output tensor of shape (N, output_size).
        """
        self.x = x
        out = xp.dot(x, self.w) + self.b
        return out

    def backward(self, grad_output):
        """
        Backward pass of the dense layer.

        Args:
            grad_output (cupy.ndarray): Gradient of the loss with respect to the output of the layer,
                                        shape (N, output_size).

        Returns:
            cupy.ndarray: Gradient with respect to the input, shape (N, input_size).
        """
        # Gradient w.r.t. weights
        self.dw = xp.dot(self.x.T, grad_output)

        # Gradient w.r.t. biases
        self.db = xp.sum(grad_output, axis=0)

        # Gradient w.r.t. input
        dx = xp.dot(grad_output, self.w.T)
        return dx
