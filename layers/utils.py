from layers.backend import xp as cp, as_strided
import time


def im2col_loop(X, kernel_size: int, stride=1, padding=0):
    """
    Convert a 4D tensor into a 2D matrix for convolution operations.

    Parameters:
    - X: Input stack of shape (N, C, H, W)
    - kernel_size: Size of the convolution kernel (int)
    - stride: Stride of the convolution (int)
    - padding: Padding added to the input (int)

    Returns:
    - A 2D matrix where each column corresponds to a sliding window of the input tensor.
    """
    N, C, H, W = X.shape

    # Calculate output dimensions
    out_height = (H + 2 * padding - kernel_size) // stride + 1
    out_width = (W + 2 * padding - kernel_size) // stride + 1

    # Apply padding
    X_padded = cp.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    # Create output matrix
    col = cp.zeros((N, C * kernel_size * kernel_size, out_height * out_width))

    for h in range(out_height):
        for w in range(out_width):
            h_start = h * stride
            w_start = w * stride
            h_end = h_start + kernel_size
            w_end = w_start + kernel_size

            col[:, :, h * out_width + w] = X_padded[
                :, :, h_start:h_end, w_start:w_end
            ].reshape(N, C * kernel_size * kernel_size)

    return col.transpose(0, 2, 1).reshape(-1, C * kernel_size * kernel_size)


def test_im2col(batch_size):
    images = cp.random.randn(batch_size, 3, 224, 224).astype(cp.float32)
    cp.cuda.Device(0).synchronize()
    start = time.time()
    y = im2col_loop(images, kernel_size=3, stride=1, padding=1)
    cp.cuda.Device(0).synchronize()
    end = time.time()
    mem = cp.get_default_memory_pool().used_bytes() / (1024**2)  # Convert bytes to MB
    print(
        f"[LOOPS]Batch {batch_size}: Time = {end - start:.4f}s, Memory = {mem:.2f} MB"
    )


def im2col(X, kernel_size, stride=1, padding=1):
    """
    Convert a 4D tensor into a 2D matrix for convolution operations using cupy.
    Parameters:
    - X: Input stack of shape (N, C, H, W)
    - kernel_size: Size of the convolution kernel (int)
    - stride: Stride of the convolution (int)
    - padding: Padding added to the input (int)
    Returns:
    - A 2D matrix where each column corresponds to a sliding window of the input tensor.
    """
    N, C, H, W = X.shape
    K = kernel_size

    # Padding
    if padding > 0:
        X = cp.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    # Strides
    s0, s1, s2, s3 = X.strides
    shape = (N, C, H_out, W_out, K, K)
    strides = (s0, s1, s2 * stride, s3 * stride, s2, s3)

    patches = as_strided(X, shape=shape, strides=strides)
    patches = patches.reshape(N, C, H_out * W_out, K * K)
    patches = patches.transpose(0, 2, 1, 3).reshape(N * H_out * W_out, C * K * K)

    return patches


def test__opt_im2col(batch_size):
    images = cp.random.randn(batch_size, 3, 224, 224).astype(cp.float32)
    cp.cuda.Device(0).synchronize()
    start = time.time()
    y = im2col(images, kernel_size=3, stride=1, padding=1)
    cp.cuda.Device(0).synchronize()
    end = time.time()
    mem = cp.get_default_memory_pool().used_bytes() / (1024**2)  # Convert bytes to MB
    print(
        f"[As_STRIDED] Batch {batch_size}: Time = {end - start:.4f}s, Memory = {mem:.2f} MB"
    )


def col2im(col, input_shape, kernel_size, stride=1, padding=0):
    """
    Convert a 2D matrix back to a 4D tensor.

    Parameters:
    - col: Input 2D matrix
    - input_shape: Shape of the original input tensor (N, C, H, W)
    - kernel_size: Size of the convolution kernel (int)
    - stride: Stride of the convolution (int)
    - padding: Padding added to the input (int)

    Returns:
    - A 4D tensor reconstructed from the 2D matrix.
    """
    N, C, H, W = input_shape
    K = kernel_size

    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = cp.zeros((N, C, H_padded, W_padded), dtype=col.dtype)

    s0, s1, s2, s3 = x_padded.strides
    shape = (N, C, H_out, W_out, K, K)
    strides = (s0, s1, s2 * stride, s3 * stride, s2, s3)

    x_strided = as_strided(x_padded, shape=shape, strides=strides)

    col_reshaped = (
        col.reshape(N, H_out * W_out, C, K * K).transpose(0, 2, 1, 3).reshape(shape)
    )
    x_strided += col_reshaped

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, layer):
        """
        Update parameters of a layer using Adam optimizer.

        Args:
            layer: Any layer object with attributes:
                - w, dw (weights and their gradients)
                - b, db (biases and their gradients)
        """
        self.t += 1

        for param_name in ["w", "b"]:
            if not hasattr(layer, param_name):
                continue

            param = getattr(layer, param_name)
            grad = getattr(layer, f"d{param_name}")
            key = f"{id(layer)}_{param_name}"
            if key not in self.m:
                self.m[key] = cp.zeros_like(param)
                self.v[key] = cp.zeros_like(param)

            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad**2)

            # Compute bias-corrected estimates
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # Update parameters
            param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)
            setattr(layer, param_name, param)


class Sequential:
    """
    Sequential container for layers.
    Applies layers in the order they are added.
    """

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def update(self, optimizer):
        """
        Updates parameters of all layers using the provided optimizer.
        """
        for layer in self.layers:
            if hasattr(layer, "w") and hasattr(layer, "dw"):
                optimizer.update(layer)


def compute_accuracy(logits, labels):
    predictions = cp.argmax(logits, axis=1)
    return cp.mean(predictions == labels)
