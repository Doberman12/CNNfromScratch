import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from layers.dropout import Dropout
from layers.backend import xp as cp


@pytest.fixture
def input_tensor():
    return cp.ones((2, 3))


def test_forward_training_mode_sets_mask(input_tensor):
    layer = Dropout(drop_prob=0.5)
    output = layer.forward(input_tensor)
    assert layer.mask is not None
    assert output.shape == input_tensor.shape


def test_forward_evaluation_mode_returns_input(input_tensor):
    layer = Dropout(drop_prob=0.5)
    layer.training = False
    output = layer.forward(input_tensor)
    cp.testing.assert_array_equal(output, input_tensor)


def test_backward_applies_mask(input_tensor):
    layer = Dropout(drop_prob=0.5)
    _ = layer.forward(input_tensor)
    grad_output = cp.ones_like(input_tensor)
    grad_input = layer.backward(grad_output)
    expected = grad_output * layer.mask
    cp.testing.assert_allclose(grad_input, expected, rtol=1e-5, atol=1e-8)


def test_backward_evaluation_mode_passthrough(input_tensor):
    layer = Dropout(drop_prob=0.5)
    layer.training = False
    grad_output = cp.ones_like(input_tensor)
    grad_input = layer.backward(grad_output)
    cp.testing.assert_array_equal(grad_input, grad_output)
