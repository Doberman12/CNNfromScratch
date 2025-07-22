from layers.backend import xp


def prepare_im2col(x, K, stride=1, padding=0):
    N, C, H, W = x.shape

    if padding > 0:
        x = xp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    cols = xp.empty((N, C, K, K, H_out, W_out), dtype=x.dtype)
    for i in range(K):
        for j in range(K):
            cols[:, :, i, j, :, :] = x[
                :, :, i : i + stride * H_out : stride, j : j + stride * W_out : stride
            ]
    return cols, H_out, W_out


def im2col_conv(x, kernel_size, stride=1, padding=0):
    cols, H_out, W_out = prepare_im2col(x, kernel_size, stride, padding)
    N, C = x.shape[:2]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(
        N * H_out * W_out, C * kernel_size * kernel_size
    )


def im2col_pool(x, kernel_size, stride=1, padding=0):
    cols, H_out, W_out = prepare_im2col(x, kernel_size, stride, padding)
    N, C = x.shape[:2]
    return (
        cols.reshape(N, C, kernel_size * kernel_size, H_out * W_out)
        .transpose(0, 1, 3, 2)
        .reshape(N * C * H_out * W_out, kernel_size * kernel_size)
    )


def prepare_col2im(input_shape, kernel_size, stride=1, padding=0):
    N, C, H, W = input_shape
    K = kernel_size

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    H_out = (H_padded - K) // stride + 1
    W_out = (W_padded - K) // stride + 1

    return N, C, H, W, H_padded, W_padded, H_out, W_out, K


def col2im_conv(cols, input_shape, kernel_size, stride=1, padding=0):
    N, C, _, _, H_p, W_p, H_out, W_out, K = prepare_col2im(
        input_shape, kernel_size, stride, padding
    )

    cols_reshaped = cols.reshape(N, H_out, W_out, C, K, K).transpose(0, 3, 4, 5, 1, 2)
    out = xp.zeros((N, C, H_p, W_p), dtype=cols.dtype)

    for i in range(K):
        for j in range(K):
            out[
                :, :, i : i + stride * H_out : stride, j : j + stride * W_out : stride
            ] += cols_reshaped[:, :, i, j, :, :]

    return (
        out[:, :, padding : H_p - padding, padding : W_p - padding]
        if padding > 0
        else out
    )


def col2im_pool(cols, input_shape, kernel_size, stride=1, padding=0):
    N, C, _, _, H_p, W_p, H_out, W_out, K = prepare_col2im(
        input_shape, kernel_size, stride, padding
    )

    cols_reshaped = cols.reshape(N, C, H_out * W_out, K * K).transpose(0, 1, 3, 2)
    cols_reshaped = cols_reshaped.reshape(N * C, K * K, H_out, W_out)

    out = xp.zeros((N * C, H_p, W_p), dtype=cols.dtype)
    idx = 0
    for i in range(K):
        for j in range(K):
            out[
                :, i : i + stride * H_out : stride, j : j + stride * W_out : stride
            ] += cols_reshaped[:, idx, :, :]
            idx += 1

    out = out.reshape(N, C, H_p, W_p)
    return (
        out[:, :, padding : H_p - padding, padding : W_p - padding]
        if padding > 0
        else out
    )


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
                self.m[key] = xp.zeros_like(param)
                self.v[key] = xp.zeros_like(param)

            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad**2)

            # Compute bias-corrected estimates
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # Update parameters
            param -= self.lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)
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
    predictions = xp.argmax(logits, axis=1)
    return xp.mean(predictions == labels)
