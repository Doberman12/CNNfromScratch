import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from layers.backend import xp
from layers import utils


def test_im2col_and_col2im_conv():
    x = xp.ones((1, 1, 5, 5), dtype=xp.float32)
    kernel_size = 3
    stride = 1
    padding = 1

    cols = utils.im2col_conv(x, kernel_size, stride, padding)
    x_reconstructed = utils.col2im_conv(cols, x.shape, kernel_size, stride, padding)

    expected_counts = xp.array(
        [
            [4, 6, 6, 6, 4],
            [6, 9, 9, 9, 6],
            [6, 9, 9, 9, 6],
            [6, 9, 9, 9, 6],
            [4, 6, 6, 6, 4],
        ],
        dtype=xp.float32,
    )

    assert xp.allclose(x_reconstructed[0, 0], expected_counts, atol=1e-5)


def test_im2col_and_col2im_pool():
    x = xp.ones((1, 1, 4, 4), dtype=xp.float32)
    kernel_size = 2
    stride = 2
    padding = 0

    cols = utils.im2col_pool(x, kernel_size, stride, padding)
    x_reconstructed = utils.col2im_pool(cols, x.shape, kernel_size, stride, padding)

    assert x_reconstructed.shape == x.shape
    assert xp.allclose(x_reconstructed, x, atol=1e-5)


def test_compute_accuracy():
    logits = xp.array([[0.1, 0.8, 0.1], [0.3, 0.2, 0.5]])
    labels = xp.array([1, 2])
    acc = utils.compute_accuracy(logits, labels)
    assert xp.isclose(acc, 1.0)


def test_adam_optimizer_update():
    class DummyLayer:
        def __init__(self):
            self.w = xp.ones((3,))
            self.dw = xp.full((3,), 0.1)
            self.b = xp.zeros((3,))
            self.db = xp.full((3,), 0.01)

    layer = DummyLayer()
    optimizer = utils.AdamOptimizer()
    w_before = layer.w.copy()
    optimizer.update(layer)
    assert not xp.allclose(layer.w, w_before)


def test_sequential_forward_backward():
    class IdentityLayer:
        def forward(self, x):
            return x

        def backward(self, grad):
            return grad

    layers_seq = utils.Sequential([IdentityLayer(), IdentityLayer()])
    x = xp.ones((2, 2))
    out = layers_seq.forward(x)
    grad = layers_seq.backward(out)

    assert xp.array_equal(out, x)
    assert xp.array_equal(grad, x)
