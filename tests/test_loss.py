import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from layers.backend import xp
from layers.softmaxcrossentropyloss import SoftmaxCrossEntropyLoss


def test_softmax_cross_entropy_forward():
    loss = SoftmaxCrossEntropyLoss()

    logits = xp.array([[2.0, 1.0, 0.1]])
    labels = xp.array([0])

    output = loss.forward(logits, labels)

    exp_logits = xp.exp(logits - xp.max(logits))
    probs = exp_logits / xp.sum(exp_logits)
    expected_loss = -xp.log(probs[xp.arange(len(labels)), labels])
    expected_loss = xp.mean(expected_loss)

    assert xp.isclose(output, expected_loss)


def test_softmax_cross_entropy_backward():
    loss = SoftmaxCrossEntropyLoss()

    logits = xp.array([[2.0, 1.0, 0.1]])
    labels = xp.array([0])

    loss.forward(logits, labels)
    grad = loss.backward()

    exp_logits = xp.exp(logits - xp.max(logits))
    probs = exp_logits / xp.sum(exp_logits)
    expected_grad = probs.copy()
    expected_grad[0, labels[0]] -= 1
    expected_grad = expected_grad / len(labels)

    assert xp.allclose(grad, expected_grad)
