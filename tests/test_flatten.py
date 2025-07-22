import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from layers.backend import xp as cp
from layers.flatten import Flatten


@pytest.fixture
def input_tensor():
    return cp.arange(24).reshape(2, 3, 4)


def test_forward_flattens_tensor(input_tensor):
    layer = Flatten()
    output = layer.forward(input_tensor)

    assert output.ndim == 2
    assert output.shape == (2, 12)


def test_backward_restores_original_shape(input_tensor):
    layer = Flatten()
    output = layer.forward(input_tensor)

    grad_output = cp.ones_like(output)
    grad_input = layer.backward(grad_output)

    assert grad_input.shape == input_tensor.shape
    cp.testing.assert_allclose(grad_input, grad_output.reshape(input_tensor.shape))


def test_forward_backward_consistency(input_tensor):
    layer = Flatten()
    flattened = layer.forward(input_tensor)
    restored = layer.backward(flattened)

    cp.testing.assert_array_equal(restored, input_tensor)
