import numpy as np


class Activation:
    def activate(self, x) -> int:
        pass

    def activate_prime(self, x) -> int:
        pass


class ReLU(Activation):
    def activate(self, x):
        return x * (x > 0)

    def activate_prime(self, x):
        return 1.0 * (x > 0)


class Hyperbolic_Tangent(Activation):
    def activate(self, x) -> int:
        return np.tanh(x)

    def activate_prime(self, x) -> int:
        return 1 - np.tanh(x) ** 2


class Sigmoid(Activation):
    def activate(self, x) -> int:
        return 1 / (1 + np.exp(-x))

    def activate_prime(self, x) -> int:
        return self.activate(x) * (1 - self.activate(x))
