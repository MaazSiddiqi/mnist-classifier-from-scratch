import numpy as np

np.random.seed(0)

def create_data(points, classes):
    # np.random.seed(0) # set seed for reproducibility

    # create data and labels
    X = np.zeros((points * classes, 2)) # data matrix (each row = single example)
    y = np.zeros(points * classes, dtype='uint8') # class labels

    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points) # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2 # theta
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number

    return X, y

X, y = create_data(100, 3)

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


class Activation_ReLU:
    def forward(self, inputs):
        # ReLU activation: f(x) = max(0, x)
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


# Create the layers, prev neurons = next inputs

# input layer = X = 3 inputs, 4 neurons
# hidden layer = 4 inputs, 5 neurons
# output layer = 5 inputs, 2 neurons

X, y = create_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
