import cupy as cp
from layers.base import Layer
from cupy.lib.stride_tricks import as_strided

class MaxPool2D(Layer):
    """Max Pooling Layer Implementation
    Applies max pooling operation over 2D spatial dimensions.
    Parameters:
    - kernel_size: Size of the pooling window (int)
    - stride: Stride of the pooling operation (int)
    """
    def __init__(self, kernel_size: int, stride: int):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """Forward pass of the max pooling layer.
        Args:
            x (cupy.ndarray): Input tensor of shape (N, C, H, W).
        Returns:
            cupy.ndarray: Output tensor of shape (N, C, H_out, W_out) after max pooling.
        """
        self.x = x
        N, C, H, W = x.shape
        KS = self.kernel_size
        S = self.stride

        # Output dimensions
        H_out = (H - KS) // S + 1
        W_out = (W - KS) // S + 1

        # Strides
        s0, s1, s2, s3 = x.strides
        shape = (N, C, H_out, W_out, KS, KS)
        strides = (s0, s1, s2 * S, s3 * S, s2, s3)

        x_strided = as_strided(x, shape=shape, strides=strides)
        x_reshaped = x_strided.reshape(N, C, H_out, W_out, -1)

        # Maximum pooling
        self.max_indices = cp.argmax(x_reshaped, axis=-1)
        out = cp.max(x_reshaped, axis=-1)
        return out

    def backward(self, grad_output):
        """Backward pass of the max pooling layer.
        Args:
            grad_output (cupy.ndarray): Gradient of the loss with respect to the output of the layer,
                                        shape (N, C, H_out, W_out).
        Returns:
            cupy.ndarray: Gradient with respect to the input, shape (N, C, H, W).
        """
        N, C, H, W = self.x.shape
        KS = self.kernel_size
        S = self.stride
        H_out = (H - KS) // S + 1
        W_out = (W - KS) // S + 1

        # Przygotuj gradient wejścia
        dx = cp.zeros_like(self.x)

        # Stride'y
        s0, s1, s2, s3 = dx.strides
        shape = (N, C, H_out, W_out, KS, KS)
        strides = (s0, s1, s2 * S, s3 * S, s2, s3)

        dx_strided = as_strided(dx, shape=shape, strides=strides)
        dx_reshaped = dx_strided.reshape(N, C, H_out, W_out, -1)

        N, C, H_out, W_out = self.max_indices.shape
        flat_indices = cp.ravel_multi_index(
            (
                cp.repeat(cp.arange(N), C * H_out * W_out),
                cp.tile(cp.repeat(cp.arange(C), H_out * W_out), N),
                cp.tile(cp.repeat(cp.arange(H_out), W_out), N * C),
                cp.tile(cp.arange(W_out), N * C * H_out),
                self.max_indices.ravel()
            ),
        dims=(N, C, H_out, W_out, self.kernel_size * self.kernel_size)
        )

        dx_reshaped = dx_strided.reshape(-1, self.kernel_size * self.kernel_size)
        dx_reshaped[...] = 0
        dx_reshaped.ravel()[flat_indices] = grad_output.ravel()


        return dx
