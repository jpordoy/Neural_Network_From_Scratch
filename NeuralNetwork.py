import numpy as np
 
class NeuralNetwork:
    '''
    The NeuralNetwork class represents a simple feedforward neural network with multiple layers.

    Attributes:
        layers (list): A list of Layer objects that make up the neural network.
        learning_rate (float): The learning rate used for updating the parameters during training.

    Methods:
        add_layer(layer): Adds a layer to the neural network.
        forward(inputs): Propagates inputs through all layers of the network and returns the final output.
        backward(targets): Propagates gradients back through all layers to update parameters.
        train(inputs, targets, learning_rate): Performs a single training iteration, including forward and backward passes.
        loss(predictions, targets): Computes the Mean Squared Error (MSE) loss between predictions and targets.
        loss_derivative(predictions, targets): Computes the derivative of the loss function with respect to the predictions.
    '''
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, targets):
        d_outputs = self.loss_derivative(self.layers[-1].outputs, targets)
        for layer in reversed(self.layers):
            d_outputs = layer.backward(d_outputs)

    def train(self, inputs, targets, learning_rate):
        self.forward(inputs)
        self.backward(targets)
        self.learning_rate = learning_rate

    def loss(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)

    def loss_derivative(self, predictions, targets):
        return 2 * (predictions - targets) / len(targets)