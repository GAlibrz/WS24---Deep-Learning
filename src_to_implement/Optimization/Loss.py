import tensorflow as tf

class CrossEntropyLoss:
    def __init__(self):
        self.label_tensor = None

    def forward(prediction_tensor, label_tensor):
        self.label_tensor = label_tensor
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction_tensor, labels=label_tensor))
    
    def backward(label_tensor):
        return tf.nn.softmax(label_tensor)