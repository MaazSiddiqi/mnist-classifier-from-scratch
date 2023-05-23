import numpy as np


class Layer_Dense:
    # initialize the weights and biases
    # best practice to initialize weights with random values between -1 and 1
    # best practice to initialize biases with 0, unless you have a good reason not to
    def __init__(self, n_inputs, n_neurons) -> None:
        # initialize the weights and biases

        # np.random.randn() returns a sample (or samples) from the "standard normal" distribution (mean=0, stdev=1), in shape of (n_inputs, n_neurons)
        # multiply by 0.1 to make the values smaller (closer to 0)
        self.weights = 0.1 * np.random.randn(n_neurons, n_inputs)

        # np.zeros() returns a new array of given shape, filled with zeros
        self.biases = np.zeros((n_neurons, 1))
        pass

    def forward(self, inputs):
        # save the inputs for later use in backpropagation
        self.inputs = inputs

        # calculate the output values from inputs, weights, and biases
        self.output = np.dot(self.weights, inputs) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        # calculate the gradient of weights, biases, and inputs

        # gradient of weights = output gradient * inputs
        weights_gradient = np.dot(output_gradient, self.inputs.T)
        self.weights -= learning_rate * weights_gradient

        # gradient of biases = output gradient
        self.biases -= learning_rate * output_gradient

        # gradient of inputs = weights * output gradient
        return np.dot(self.weights.T, output_gradient)


class Activation_ReLU:
    def forward(self, inputs):
        # save the inputs for later use in backpropagation
        self.inputs = inputs

        # ReLU activation: f(x) = max(0, x)
        # ReLU derivative: f'(x) = 1 if x > 0, otherwise 0
        self.output = np.maximum(0, inputs)
        self.derivative = lambda x: 1 if x > 0 else 0

    def backward(self, output_gradient):
        # calculate the gradient of inputs
        # gradient of inputs = output gradient * derivative of inputs
        return np.multiply(output_gradient, self.derivative(self.inputs))


class Activation_Softmax:
    def forward(self, inputs):
        # Exponentiate to make all values positive without losing relative meaning
        # Subtract the max value to make the values smaller and prevent overflow, retains relative meaning
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize the values to be between 0 and 1
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, output_gradient):
        pass


class Loss:
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            # Passed in scalars
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            # Passed in one-hot vectors
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        # Calculate loss
        # E = 1/n * sum((y_true - y_pred)^2)
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_pred, y_true):
        # Calculate gradient
        # dE/dy_pred = 2 * (y_pred - y_true) / n
        return 2 * (y_pred - y_true) / np.size(y_true)


# Create the layers, prev neurons = next inputs

# input layer = X = 3 inputs, 4 neurons
# hidden layer = 4 inputs, 5 neurons
# output layer = 5 inputs, 2 neurons
