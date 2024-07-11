import numpy as np

class Neuron:
    '''
    The Neuron class represents a single neuron in a neural network. 

    Attributes:
        weights (numpy.ndarray): The weights associated with the inputs of the neuron.
        bias (float): The bias term added to the weighted sum of inputs.
        inputs (numpy.ndarray): The inputs to the neuron, stored for use in backpropagation.
        dot_product (float): The result of the weighted sum of inputs plus the bias, before applying the activation function.
        output (float): The output of the neuron after applying the activation function.

    Methods:
        forward(inputs): Computes the weighted sum of inputs plus the bias, applies an activation function, and returns the output.
        relu(x): Applies the ReLU activation function.
        relu_derivative(x): Computes the derivative of the ReLU activation function.
        backward(d_output): Computes the gradients of the loss with respect to the neuron's inputs, weights, and bias.
        update_parameters(d_weights, d_bias, learning_rate): Updates the weights and bias using the computed gradients and the specified learning rate.
    '''
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()

    def forward(self, inputs):
        self.inputs = inputs  # Store inputs for use in backpropagation
        self.dot_product = np.dot(inputs, self.weights) + self.bias
        self.output = self.relu(self.dot_product)
        return self.output

    def relu(self, x):
        return max(0, x)

    def relu_derivative(self, x):
        return 1 if x > 0 else 0

    def backward(self, d_output):
        d_relu = d_output * self.relu_derivative(self.dot_product)
        d_weights = self.inputs * d_relu
        d_bias = d_relu
        d_inputs = self.weights * d_relu
        return d_inputs, d_weights, d_bias

    def update_parameters(self, d_weights, d_bias, learning_rate):
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
