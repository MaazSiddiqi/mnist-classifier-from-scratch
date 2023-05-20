import numpy as np

np.random.seed(0)

# Input Set
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
] # shape: 3,4,1

class Layer_Dense:
    # initialize the weights and biases
    # best practice to initialize weights with random values between -1 and 1
    # best practice to initialize biases with 0, unless you have a good reason not to
    def __init__(self, n_inputs, n_neurons) -> None:
        # initialize the weights and biases

        # np.random.randn() returns a sample (or samples) from the "standard normal" distribution (mean=0, stdev=1), in shape of (n_inputs, n_neurons)
        # multiply by 0.1 to make the values smaller (closer to 0)
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)

        # np.zeros() returns a new array of given shape, filled with zeros
        self.biases = np.zeros((1, n_neurons))
        pass

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        pass


# Create the layers, prev neurons = next inputs

# input layer = X = 3 inputs, 4 neurons
# hidden layer = 4 inputs, 5 neurons
# output layer = 5 inputs, 2 neurons

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)

# print(layer1.output)

layer2.forward(layer1.output)

print(layer2.output)
