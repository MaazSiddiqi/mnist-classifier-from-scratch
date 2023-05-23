from layers import Dense, Tanh_Activation, ReLU_Activation
from loss import Mean_Squared_Error
from network import Network
import numpy as np

# Network Sanity check for XOR problem

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

print("X: ", X)
print("Y: ", Y)

layers = [Dense(2, 2), ReLU_Activation(), Dense(2, 1), Tanh_Activation()]
mse = Mean_Squared_Error()

network = Network(layers, mse)

network.train(X, Y, epochs=10000, learning_rate=0.1, log=True)
network.test(X, Y)

# epochs = 10000
# learning_rate = 0.1

# for epoch in range(epochs):
#     error = 0

#     for x, y in zip(X, Y):
#         # Forward Propagation
#         outputs = x
#         outputs = predict(outputs, network)

#         # Calculate error
#         error += mse.calculate(outputs, y)

#         # Backward Propagation
#         gradient = mse.prime(outputs, y)
#         for layer in reversed(network):
#             gradient = layer.backward(gradient, learning_rate)

#     error /= len(X)

#     print(f"{epoch + 1}/{epochs}, error={error}")
