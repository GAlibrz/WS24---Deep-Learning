import numpy as np
import tensorflow as tf
from Base import BaseLayer
from Optimization import Optimizers

class FullyConnected(BaseLayer):


    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.input = None
        # + 1 is added here to represent for the biases, additional care is needed therefor in forward and bachward paths
        self.weights = np.random.rand(output_size, input_size)
        self.biases = np.zeros((output_size, 1))
        
        self._optimizer = None

    def forward(self, input_tensor):
        self.input = input_tensor #needed in backward path
        weights_transpose = np.transpose(self.weights)
        result_without_biases = tf.matmul(weights_transpose, input_tensor)
        result_with_bias = tf.add(result_without_biases, self.biases)
        return result_without_biases

        @property
        def optimizer(self):
            return self._optimizer
        
        @optimizer.setter
        def optimizer(self, optimizer):
            self._optimizer = optimizer
            return self._optimizer
        
    def backward(self, error_tensor):
        self.gradien_weights_without_biases = tf.matmul(error_tensor, self.input.T)
        self.biases_gradient = tf.ones(self.output_size)
        self._gradient_weights =  tf.concat([self.gradien_weights_without_biases, self.biases_gradient[tf.newaxis, :]], axis=-1)
        self.weights = self._optimizer.calculate_update(self.weights, self.gradien_weights)
    
    @property
    def gradien_weights(self):
        return self._gradient_weights  
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self._optimizer:
