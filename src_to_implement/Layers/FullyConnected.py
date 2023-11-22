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
        self.weights = np.random.rand(output_size, input_size )
        self.biases = np.zeros((output_size, 1))
        
        self._optimizer = None


        #print("i do not have the time")

    def forward(self, input_tensor):
        self.input = input_tensor #needed in backward path
        #weights_transpose = np.transpose(self.weights)
        result_without_biases = tf.matmul(input_tensor, self.weights)
        result_with_biases = result_without_biases + self.biases
        #resutl_of_biases = tf.matmul()
        #result_with_bias = tf.add(result_without_biases, self.biases)
        return result_with_biases


    @property
    def optimizer(self):
        return self._optimizer
        
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        return self._optimizer
    @property
    def gradien_weights(self):
        return self._gradient_weights  
    
        
    def backward(self, error_tensor):
        self.gradien_weights_without_biases = tf.matmul(error_tensor, self.input.T)
        self.biases_gradient = tf.ones(self.output_size)
        self._gradient_weights =  tf.concat([self.gradien_weights_without_biases, self.biases_gradient[tf.newaxis, :]], axis=-1)
        self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
    