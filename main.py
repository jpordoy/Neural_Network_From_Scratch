from Layer import Layer
from NeuralNetwork import NeuralNetwork
import numpy as np

def main():
    # Create a neural network with 3 layers: input, hidden, output
    input_layer = Layer(num_neurons=4, num_inputs_per_neuron=2)
    hidden_layer = Layer(num_neurons=3, num_inputs_per_neuron=4)
    output_layer = Layer(num_neurons=1, num_inputs_per_neuron=3)

    neural_network = NeuralNetwork()
    neural_network.add_layer(input_layer)
    neural_network.add_layer(hidden_layer)
    neural_network.add_layer(output_layer)

    # Example training data
    example_input = np.array([1, 2])
    example_target = np.array([0])

    # Train the network
    learning_rate = 0.01
    for epoch in range(1000):
        output = neural_network.forward(example_input)
        neural_network.train(example_input, example_target, learning_rate)
        loss = neural_network.loss(output, example_target)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    # Perform forward pass after training
    final_output = neural_network.forward(example_input)
    print("Final output:", final_output)

if __name__ == "__main__":
    main()
