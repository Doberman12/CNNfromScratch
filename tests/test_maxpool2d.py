import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from layers.maxpool2d import MaxPool2D
from layers.backend import xp


def test_forward_maxpool2d():
    x = xp.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]
    )  # shape: (1, 1, 4, 4)

    pool = MaxPool2D(kernel_size=2, stride=2)
    output = pool.forward(x)

    expected = xp.array([[[[6, 8], [14, 16]]]])  # shape: (1, 1, 2, 2)

    assert xp.allclose(output, expected), f"Expected {expected}, got {output}"


def test_shape_after_forward():
    x = xp.ones((2, 3, 8, 8))
    pool = MaxPool2D(kernel_size=2, stride=2)
    out = pool.forward(x)
    assert out.shape == (2, 3, 4, 4), f"Expected shape (2, 3, 4, 4), got {out.shape}"


def test_shape_after_backward():
    x = xp.random.rand(2, 2, 6, 6)
