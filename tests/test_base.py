import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from layers.base import Layer
def test_layer_name_default():
    layer = Layer()
    assert layer.name == "Layer"

def test_layer_name_custom():
    layer = Layer(name="CustomLayer")
    assert layer.name == "CustomLayer"

def test_forward_not_implemented():
    layer = Layer()
    with pytest.raises(NotImplementedError, match="Forward method not implemented."):
        layer.forward()

def test_backward_not_implemented():
    layer = Layer()
    with pytest.raises(NotImplementedError, match="Backward method not implemented."):
        layer.backward()
