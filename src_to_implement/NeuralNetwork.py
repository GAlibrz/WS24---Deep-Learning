from typing import List, Any
import copy 



class NeuralNetwork:
    def __init__(self, optimizer, loss: List[Any] = None, layers: List[Any] = None, data_layer = None, loss_layer = None):
        self.optimizer = optimizer
        self.loss = loss
        self.layers = layers
        self.data_layer = data_layer
        self.loss_layer = loss_layer

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        curr_output = self.input_tensor
        for layer in self.layers:
            curr_output = layer.forward(curr_output)
        return curr_output
    
    def backward(self):
        curr_loss = self.label_tensor
        for layer in reversed(self.layers):
            curr_loss = layer.backward(curr_loss)
        return curr_loss

    def append_layer(self, layer):
        if layer.trainable:
            self.optimizer_copy = copy.deepcopy(self.optimizer)
            layer.optimizer = self.optimizer_copy
        self.layers.append(layer)

    def train(self, iterations):
        self.iterational_losses = []
        for i in iterations:
            self.forward()
            
            

