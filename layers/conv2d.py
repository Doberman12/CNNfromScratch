from layers.base import Layer
from layers.backend import xp
from layers.utils import im2col_conv, col2im_conv

"""
Conv2D Layer Implementation
"""


class Conv2D(Layer):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.w = (
            xp.random.randn(output_channels, input_channels, kernel_size, kernel_size)
            * 0.01
        )  # Initialize weights
        self.b = xp.zeros(output_channels)  # Initialize biases

    def forward(self, x):
        """Forward pass of the convolutional layer.
        Args:
            x (cupy.ndarray): Input tensor of shape (N, C_in, H, W).
        Returns:
            cupy.ndarray: Output tensor of shape (N, C_out, H_out, W_out).
        """
        self.x = x
        N, C_in, H, W = x.shape
        x_col = im2col_conv(x, self.kernel_size, self.stride, self.padding)
        w_col = self.w.reshape(self.output_channels, -1)
        out = xp.dot(x_col, w_col.T) + self.b

        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        out = out.reshape(N, H_out, W_out, self.output_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, grad_output):
        """
        Backward pass of the convolutional layer.

        Args:
            grad_output (xp.ndarray): Gradient of the loss with respect to the output of the layer,
                                    shape (N, C_out, H_out, W_out)
        Returns:
            xp.ndarray: Gradient with respect to the input, shape (N, C_in, H, W)
        """
        N, C_out, H_out, W_out = grad_output.shape

        # Reshape grad_output to (N * H_out * W_out, C_out)
        grad_output_reshaped = grad_output.transpose(0, 2, 3, 1).reshape(-1, C_out)

        # im2col of input from forward pass
        x_col = im2col_conv(
            self.x, self.kernel_size, self.stride, self.padding
        )  # shape: (N * H_out * W_out, C_in * K * K)

        # Gradient w.r.t. weights
        dw = xp.dot(grad_output_reshaped.T, x_col)
        dw = dw.reshape(self.w.shape)

        # Gradient w.r.t. biases
        db = grad_output_reshaped.sum(axis=0)

        # Gradient w.r.t. input
        w_col = self.w.reshape(C_out, -1)
        dx_col = xp.dot(
            grad_output_reshaped, w_col
        )  # shape: (N * H_out * W_out, C_in * K * K)
        dx = col2im_conv(
            dx_col, self.x.shape, self.kernel_size, self.stride, self.padding
        )

        # Save gradients
        self.dw = dw / self.x.shape[0]  # Average over batch size
        self.db = db / self.x.shape[0]

        return dx
