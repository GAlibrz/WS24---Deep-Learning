import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        super().__init__()
        self.prediction_store = None

    def forward(self, input_tensor, label_tensor):
        self.prediction_store = input_tensor
        return np.sum(-np.log(input_tensor[label_tensor == 1] + np.finfo(float).eps))

    def backward(self, label_tensor):
        return -np.divide(label_tensor, self.prediction_store)
