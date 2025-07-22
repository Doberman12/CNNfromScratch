import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from layers.maxpool2d import MaxPool2D
from layers.backend import xp as cp, as_strided


def test_forward_maxpool2d():
    x = cp.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]
    )  # shape: (1, 1, 4, 4)

    pool = MaxPool2D(kernel_size=2, stride=2)
    output = pool.forward(x)

    expected = cp.array([[[[6, 8], [14, 16]]]])  # shape: (1, 1, 2, 2)

    assert cp.allclose(output, expected), f"Expected {expected}, got {output}"


def test_backward_maxpool2d():
    x = cp.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]
    )  # shape: (1, 1, 4, 4)

    pool = MaxPool2D(kernel_size=2, stride=2)
    out = pool.forward(x)

    grad_output = cp.array([[[[1, 2], [3, 4]]]])  # shape: (1, 1, 2, 2)

    dx = pool.backward(grad_output)

    # Sprawdź tylko wartości w miejscach maksymalnych — bo tylko tam gradient powinien być różny od zera
    nonzero_indices = cp.argwhere(dx != 0)
    nonzero_values = dx[dx != 0]

    # Wartości maxów z forward:
    expected_positions = cp.array(
        [
            [0, 0, 1, 1],  # 6
            [0, 0, 1, 3],  # 8
            [0, 0, 3, 1],  # 14
            [0, 0, 3, 3],  # 16
        ]
    )

    expected_values = cp.array([1, 2, 3, 4])

    assert (
        nonzero_indices.shape == expected_positions.shape
    ), f"Oczekiwano {expected_positions.shape[0]} niezerowych, ale było {nonzero_indices.shape[0]}"

    for idx, val in zip(expected_positions, expected_values):
        actual_val = dx[tuple(idx)]
        assert (
            actual_val == val
        ), f"Gradient w {idx} powinien być {val}, ale jest {actual_val}"


def test_shape_after_forward():
    x = cp.ones((2, 3, 8, 8))
    pool = MaxPool2D(kernel_size=2, stride=2)
    out = pool.forward(x)
    assert out.shape == (2, 3, 4, 4), f"Expected shape (2, 3, 4, 4), got {out.shape}"


def test_shape_after_backward():
    x = cp.random.rand(2, 2, 6, 6)
