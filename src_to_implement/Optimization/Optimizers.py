class Sgd:
  learning_rate = 1.00
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate
  def calculate_update(self, weight_tensor, gradient_tensor):
    updated_weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
    return updated_weight_tensor