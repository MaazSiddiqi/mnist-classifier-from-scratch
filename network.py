import numpy as np


class Network:
    def __init__(self, layers, loss_func) -> None:
        self.layers = layers
        self.loss_func = loss_func

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self) -> str:
        return str(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, learning_rate, log=False):
        for epoch in range(epochs):
            error = 0

            for x, y in zip(x_train, y_train):
                # Forward Propagation
                outputs = self.predict(x)

                # Calculate error
                error += self.loss_func.calculate(outputs, y)

                # Backward Propagation
                gradient = self.loss_func.prime(outputs, y)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)

            error /= len(x_train)

            if log:
                print(
                    f"{epoch + 1}/{epochs}, error={error} learning_rate={learning_rate}"
                )

    def test(self, x_test, y_test, log=False):
        correct = 0
        for x, y in zip(x_test, y_test):
            outputs = self.predict(x)

            if np.argmax(outputs) == np.argmax(y):
                correct += 1

        if log:
            print(f"Test accuracy: {correct / len(x_test)}")

        return correct / len(x_test)
