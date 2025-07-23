class Layer:
    def __init__(self, name: str = None):
        """
        Initialize the layer with an optional name.

        Args:
            name (str): Optional name for the layer.
        """
        self.name = name if name else self.__class__.__name__

    def forward(self, *args, **kwargs):
        """Forward pass of the layer. To be implemented by subclasses."""
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, *args, **kwargs):
        """Backward pass of the layer. To be implemented by subclasses."""
        raise NotImplementedError("Backward method not implemented.")
