import numpy as np


class Activation:
    def activate(self, x) -> int:
        pass

    def activate_prime(self, x) -> int:
        pass


class ReLU(Activation):
    def activate(self, x):
        return x if x > 0 else 0

    def activate_prime(self, x):
        return 1 if x > 0 else 0


class Hyperbolic_Tangent(Activation):
    def activate(self, x) -> int:
        return np.tanh(x)

    def activate_prime(self, x) -> int:
        return 1 - np.tanh(x) ** 2
