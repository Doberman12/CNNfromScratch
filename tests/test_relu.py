import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from layers.backend import xp
from layers.ReLU import ReLU

def test_relu_forward():
    relu = ReLU()
    input_array = xp.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    expected_output = xp.array([0.0, 0.0, 0.0, 1.0, 3.0])

    output = relu.forward(input_array)
    assert xp.array_equal(output, expected_output)


def test_relu_backward():
    relu = ReLU()
    input_array = xp.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    grad_output = xp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    relu.forward(input_array)

    expected_grad = xp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    grad_input = relu.backward(grad_output)
    assert xp.array_equal(grad_input, expected_grad)
