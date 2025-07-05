from layers.base import Layer
import cupy as cp
class SoftmaxCrossEntropyLoss(Layer):
    """Softmax Cross Entropy Loss Layer
    Computes the softmax cross-entropy loss between logits and labels.
    This layer applies the softmax function to the logits and computes the cross-entropy loss.
    """
    def forward(self, logits, labels):
        """        Forward pass of the softmax cross-entropy loss layer.
        Args:   
            logits (cupy.ndarray): Logits from the previous layer, shape (N, C).
            labels (cupy.ndarray): True labels, shape (N,).     
        Returns:
            cupy.ndarray: Computed loss value.
        """
        self.labels = labels
        logits_shifted = logits - cp.max(logits, axis=1, keepdims=True)
        exp_logits = cp.exp(logits_shifted)
        self.probs = exp_logits / cp.sum(exp_logits, axis=1, keepdims=True)

        log_likelihood = -cp.log(self.probs[cp.arange(len(labels)), labels])
        return cp.mean(log_likelihood)

    def backward(self):
        """Backward pass of the softmax cross-entropy loss layer.
        Computes the gradient of the loss with respect to the logits.   
        Returns:
            cupy.ndarray: Gradient of the loss with respect to the logits, shape (N, C).
        """
        grad = self.probs.copy()
        grad[cp.arange(len(self.labels)), self.labels] -= 1
        return grad / len(self.labels)
