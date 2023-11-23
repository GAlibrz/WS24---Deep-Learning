class BaseLayer:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):

        raise NotImplementedError("Forward pass must be implemented in derived classes.")

    def backward(self, error_tensor):

        raise NotImplementedError("Backward pass must be implemented in derived classes.")
