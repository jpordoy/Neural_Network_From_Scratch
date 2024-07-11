import numpy as np
from Neuron import Neuron

class Layer:
    '''
    The Layer class represents a layer in a neural network, composed of multiple neurons.

    Attributes:
        neurons (list): A list of Neuron objects that make up the layer.
        inputs (numpy.ndarray): The inputs to the layer, stored for use in backpropagation.
        outputs (numpy.ndarray): The outputs of all neurons in the layer after the forward pass.

    Methods:
        forward(inputs): Computes the outputs of all neurons in the layer given the inputs.
        backward(d_outputs): Computes the gradients for each neuron in the layer and updates their parameters.
    '''


    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

    def forward(self, inputs):
        self.inputs = inputs  # Store inputs for use in backpropagation
        self.outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.outputs

    def backward(self, d_outputs):
        d_inputs = np.zeros(self.inputs.shape)
        for i, neuron in enumerate(self.neurons):
            d_input, d_weights, d_bias = neuron.backward(d_outputs[i])
            neuron.update_parameters(d_weights, d_bias, 0.01)
            d_inputs += d_input
        return d_inputs
