import numpy as np
import Base

class FullyConnected(Base):


    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        # + 1 is added here to represent for the biases, additional care is needed therefor in forward and bachward paths
        self.weights = np.random.rand(output_size, input_size + 1)
        
        self._optimizer = None

        @property
        def optimizer(self):
            return self._optimizer
        
        @optimizer.setter
        def optimizer(self, optimizer):
            self._optimizer = optimizer
            return self._optimizer
        
