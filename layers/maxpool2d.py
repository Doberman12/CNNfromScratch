from layers.base import Layer
from layers.backend import xp as cp
from layers.utils import im2col_pool, col2im_pool


class MaxPool2D(Layer):
    """Max Pooling Layer using im2col and col2im.
    Performs max pooling over 2D spatial dimensions using efficient tensor ops.
    """

    def __init__(self, kernel_size: int, stride: int):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        """
        Forward pass of max pooling using im2col.
        Args:
            x (xp.ndarray): Input tensor of shape (N, C, H, W)
        Returns:
            xp.ndarray: Output tensor of shape (N, C, H_out, W_out)
        """
        self.input_shape = x.shape
        K = self.kernel_size
        S = self.stride

        x_col = im2col_pool(x, kernel_size=K, stride=S, padding=0)
        self.max_indices = cp.argmax(x_col, axis=1)
        out = cp.max(x_col, axis=1)

        N, C, H, W = x.shape
        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1
        out = out.reshape(N, C, H_out, W_out)
        return out

    def backward(self, grad_output):
        """
        Backward pass of max pooling using col2im.
        Args:
            grad_output (xp.ndarray): Gradient w.r.t. output, shape (N, C, H_out, W_out)
        Returns:
            xp.ndarray: Gradient w.r.t. input, shape (N, C, H, W)
        """
        K = self.kernel_size
        S = self.stride
        N, C, H, W = self.input_shape
        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        grad_flat = grad_output.reshape(N * C * H_out * W_out)
        dx_col = cp.zeros((grad_flat.shape[0], K * K), dtype=grad_output.dtype)
        dx_col[cp.arange(dx_col.shape[0]), self.max_indices] = grad_flat

        dx = col2im_pool(
            dx_col, input_shape=self.input_shape, kernel_size=K, stride=S, padding=0
        )
        return dx
