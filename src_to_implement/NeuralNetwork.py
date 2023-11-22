from typing import List, Any



class NeuralNetwork:
    def __init__(self, optimizer, loss: List[Any], layers: List[Any], data_layer, loss_layer):
        self.optimizer = optimizer
        self.loss = loss
        self.layers = layers
        self.data_layer = data_layer
        self.loss_layer = loss_layer