import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from layers.dense import Dense
from layers.backend import xp


@pytest.fixture
def dense_layer():
    return Dense(input_size=4, output_size=3)


@pytest.fixture
def input_tensor():
    return xp.array([[1.0, 2.0, 3.0, 4.0]])


def test_forward_output_shape(dense_layer, input_tensor):
    output = dense_layer.forward(input_tensor)
    assert output.shape == (1, 3), "Output shape should be (N, output_size)"


def test_forward_computation_consistency(dense_layer, input_tensor):
    expected = xp.dot(input_tensor, dense_layer.w) + dense_layer.b
    actual = dense_layer.forward(input_tensor)
    xp.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_backward_output_shape(dense_layer, input_tensor):
    dense_layer.forward(input_tensor)
    grad_output = xp.ones((1, 3))
    grad_input = dense_layer.backward(grad_output)
    assert grad_input.shape == (1, 4), "Backward output shape should be (N, input_size)"


def test_backward_gradient_values(dense_layer, input_tensor):
    dense_layer.forward(input_tensor)
    grad_output = xp.array([[1.0, 2.0, 3.0]])
    expected_dw = xp.dot(input_tensor.T, grad_output)
    expected_db = xp.sum(grad_output, axis=0)

    xp.testing.assert_allclose(dense_layer.dw, expected_dw, rtol=1e-5, atol=1e-8)
    xp.testing.assert_allclose(dense_layer.db, expected_db, rtol=1e-5, atol=1e-8)
