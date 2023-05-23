import numpy as np

np.random.seed(0)


def create_data(points, classes):
    # np.random.seed(0) # set seed for reproducibility

    # create data and labels
    X = np.zeros((points * classes, 2))  # data matrix (each row = single example)
    y = np.zeros(points * classes, dtype="uint8")  # class labels

    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, points)
            + np.random.randn(points) * 0.2
        )  # theta
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
        # save the inputs for later use in backpropagation
        self.inputs = inputs

        # calculate the output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.inputs.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return np.dot(output_gradient, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        # ReLU activation: f(x) = max(0, x)
        self.output = np.maximum(0, inputs)

    def backward(self, output_gradient):
        # ReLU derivative: f'(x) = 1 if x > 0, otherwise 0
        # return np.multiply(output_gradient, np.heaviside(self.output, 0))
        pass


class Activation_Softmax:
    def forward(self, inputs):
        # Exponentiate to make all values positive without losing relative meaning
        # Subtract the max value to make the values smaller and prevent overflow, retains relative meaning
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize the values to be between 0 and 1
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


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
        return np.mean(np.power(y_true - y_pred, 2))


# Create the layers, prev neurons = next inputs

# input layer = X = 3 inputs, 4 neurons
# hidden layer = 4 inputs, 5 neurons
# output layer = 5 inputs, 2 neurons

X, y = create_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)
