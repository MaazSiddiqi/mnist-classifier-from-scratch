import numpy as np


class Loss:
    # y_pred = (j x 1) vector of outputs from network
    # y_true = (j x 1) vector of correct expected outputs
    def calculate(self, y_pred, y_true):
        pass

    def prime(self, y_pred, y_true):
        pass


class Mean_Squared_Error(Loss):
    def calculate(self, y_pred, y_true):
        return np.mean(np.power((y_pred - y_true), 2))

    def prime(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / np.size(y_true)
