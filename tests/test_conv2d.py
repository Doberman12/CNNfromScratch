import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from layers.conv2d import Conv2D
from layers.backend import xp as cp


print(f"USE_CPU = {os.getenv('USE_CPU')}")
print(f"Backend xp: {xp.__name__}")


@pytest.fixture
def sample_data():
    batch_size = 2
    input_channels = 3
    output_channels = 4
    height = 5
    width = 5
    kernel_size = 3
    stride = 1
    padding = 1

    x = cp.random.randn(batch_size, input_channels, height, width)
    grad_output = cp.random.randn(batch_size, output_channels, height, width)

    layer = Conv2D(input_channels, output_channels, kernel_size, stride, padding)

    return layer, x, grad_output


def test_forward_shape(sample_data):
    print(f"USE_CPU = {os.getenv('USE_CPU')}")
    print(f"Backend xp: {xp.__name__}")
    layer, x, _ = sample_data
    out = layer.forward(x)
    assert out.shape == (x.shape[0], layer.output_channels, x.shape[2], x.shape[3])


def test_backward_shape(sample_data):
    layer, x, grad_output = sample_data
    layer.forward(x)
    dx = layer.backward(grad_output)
    assert dx.shape == x.shape


def test_grad_weights_shape(sample_data):
    layer, x, grad_output = sample_data
    layer.forward(x)
    layer.backward(grad_output)
    assert layer.dw.shape == layer.w.shape


def test_grad_bias_shape(sample_data):
    layer, x, grad_output = sample_data
    layer.forward(x)
    layer.backward(grad_output)
    assert layer.db.shape == layer.b.shape


def test_forward_no_nan(sample_data):
    layer, x, _ = sample_data
    out = layer.forward(x)
    assert not cp.isnan(out).any()


def test_backward_no_nan(sample_data):
    layer, x, grad_output = sample_data
    layer.forward(x)
    dx = layer.backward(grad_output)
    assert not cp.isnan(dx).any()
