import numpy as np
from activations import ReLU, Hyperbolic_Tangent


class Layer:
    def forward(self, inputs):
        pass

    def backward(self, outputs_gradient, learning_rate):
        pass


class Dense(Layer):
    # Optimizer = Gradient Descent

    def __init__(self, inputs_size, outputs_size) -> None:
        # input_size = i, output_size = j = # nuerons
        # Initialize weights to random (j x i) matrix W
        self.weights = np.random.randn(outputs_size, inputs_size)

        # Initialize biases to 0s in a (j x 1) vector B
        self.biases = np.zeros((outputs_size, 1))

        super().__init__()

    def forward(self, inputs):
        # store inputs (X) for later use in back propogation
        self.inputs = inputs

        # Y = W * X + B
        # Y = (j x i) * (i x 1) + (j x 1) = (j x 1)
        self.outputs = np.dot(self.weights, inputs) + self.biases

        return self.outputs

    def backward(self, outputs_gradient, learning_rate):
        # dC/dw = dy/dw * dC/dy = X * outputs_gradient
        # outputs_gradient = dC/dy => (j x 1) vector
        # X = dy/dw => (i x 1)
        # dC/dw = outputs_gradient * X.T => (j x 1) * (1 x i) = (j x i) matrix
        weights_gradient = np.dot(outputs_gradient, self.inputs.T)

        # dC/db = dy/db * dC/dy
        # dy/db = 1 => b is a constant
        # dC/db = dC/dy
        bias_gradient = outputs_gradient

        # dC/dx = dy/dx * dC/dy
        # dy/dx = W => (j x i) matrix
        # dC/dy = outputs_gradient => (j x 1) vector
        # dC/dx = W.T * outputs_gradient => (i x j) * (j x 1) = (i x 1)
        # want to get (i x 1) since recursion expects a (j x 1) vector
        # i of this layer = j of previous layer
        input_gradient = np.dot(self.weights.T, outputs_gradient)

        # adjust weights and biases according to gradient
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * bias_gradient

        return input_gradient


class Activation_Layer(Layer):
    def __init__(self, activate, activate_prime) -> None:
        # assign activation function and its derivative
        self.activate = activate
        self.derivative = activate_prime

        super().__init__()

    def forward(self, inputs):
        # inputs = X = (i x 1)
        # save inputs for use in backpropogation
        self.inputs = inputs

        # element-wise multiply activation function
        return self.activate(self.inputs)

    def backward(self, outputs_gradient, learning_rate):
        # undo activation so dense layer can update weights
        # no weights or biases to update, just calculate and return dC/dx

        # dC/dx = dy/dx * dC/dy
        # dy/dx = self.derivative
        # dC/dy = outputs_gradient

        return np.multiply(outputs_gradient, self.derivative(self.inputs))


class ReLU_Activation(Activation_Layer):
    def __init__(self) -> None:
        relu = ReLU()

        super().__init__(relu.activate, relu.activate_prime)


class Tanh_Activation(Activation_Layer):
    def __init__(self) -> None:
        tanh = Hyperbolic_Tangent()

        super().__init__(tanh.activate, tanh.activate_prime)


class Softmax_Activation(Activation_Layer):
    def __init__(self) -> None:
        pass

    def forward(self, inputs):
        self.outputs = np.exp(inputs) / np.sum(np.exp(inputs), axis=0)
        return self.outputs

    def backward(self, outputs_gradient, learning_rate):
        # dC/dx = M * (I - M.T) (dot) dC/dy
        # M = np.tile(self.outputs, np.size(self.outputs)) => (n x n) matrix of repeated column vectors of outputs
        n = np.size(self.outputs)
        M = np.tile(self.outputs, n)

        return np.dot(M * (np.identity(n) - M.T), outputs_gradient)
